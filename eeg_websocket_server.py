import argparse
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue

import pandas as pd
from aiohttp import web, WSMsgType
from eeg_data_collection import stream_openbci_band_power_udp
from eeg_svm_pipeline import load_model, predict_focus

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ServerState:
    start_time: float = field(default_factory=time.time)
    last_window_index: int = 0
    connected_clients: set = field(default_factory=set)
    model_path: str = ""
    recv_port: int = 0
    ws_host: str = ""
    ws_port: int = 0
    threshold: float = 0.5
    num_channels: int = 4
    status: str = "starting"
    stop_event: threading.Event = field(default_factory=threading.Event)


def udp_stream_worker(data_queue, state, port, num_channels, band_names, timeout, max_windows):
    """Run the UDP stream in a background thread and enqueue completed windows."""
    logger.info("UDP stream worker starting on port %d", port)
    for row in stream_openbci_band_power_udp(
        port=port,
        num_channels=num_channels,
        band_names=band_names,
        timeout=timeout,
        max_windows=max_windows,
    ):
        if state.stop_event.is_set():
            logger.info("UDP stream worker received stop event")
            break
        data_queue.put(row)

    data_queue.put(None)
    logger.info("UDP stream worker finished")


async def broadcast_loop(data_queue, model, threshold, state):
    """Broadcast predictions to all connected WebSocket clients."""
    state.status = "running"
    loop = asyncio.get_running_loop()

    while not state.stop_event.is_set():
        row = await loop.run_in_executor(None, data_queue.get)
        if row is None:
            logger.info("No more UDP data available, ending broadcast loop")
            break

        state.last_window_index += 1
        df = pd.DataFrame([row])
        prediction = predict_focus(model, df, threshold=threshold).iloc[0]

        payload = {
            "type": "focus_update",
            "window_index": state.last_window_index,
            "focus": prediction["focus"],
            "confidence": float(prediction["confidence"]),
            "features": row,
            "sent_at": time.time(),
        }

        message = json.dumps(payload)
        logger.debug(
            "Broadcasting window %d to %d clients",
            state.last_window_index,
            len(state.connected_clients),
        )

        stale = []
        for ws in list(state.connected_clients):
            try:
                await ws.send_str(message)
            except Exception:
                stale.append(ws)

        for ws in stale:
            state.connected_clients.discard(ws)

    state.status = "stopped"


async def websocket_handler(request):
    """Accept connections and keep a client in the broadcast list."""
    state: ServerState = request.app["state"]
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    state.connected_clients.add(ws)
    logger.info("Client connected: %s", request.remote)
    await ws.send_json({"type": "status", "message": "connected"})

    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                logger.warning("WebSocket error from %s: %s", request.remote, ws.exception())
                break
    finally:
        state.connected_clients.discard(ws)
        logger.info("Client disconnected: %s", request.remote)

    return ws


async def status_handler(request):
    state: ServerState = request.app["state"]
    uptime_seconds = time.time() - state.start_time
    return web.json_response(
        {
            "status": "ok",
            "server_time": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": round(uptime_seconds, 1),
            "model_path": state.model_path,
            "recv_port": state.recv_port,
            "ws_host": state.ws_host,
            "ws_port": state.ws_port,
            "threshold": state.threshold,
            "window_index": state.last_window_index,
            "connected_clients": len(state.connected_clients),
            "state": state.status,
        }
    )


async def schema_handler(request):
    return web.json_response(
        {
            "websocket_path": "/ws",
            "message_types": {
                "status": {
                    "type": "status",
                    "message": "connected",
                },
                "focus_update": {
                    "type": "focus_update",
                    "window_index": "integer",
                    "focus": "focused|unfocused",
                    "confidence": "0.0-1.0",
                    "features": "object",
                    "sent_at": "unix timestamp",
                },
            },
            "control_endpoint": "/control",
            "status_endpoint": "/status",
        }
    )


async def control_handler(request):
    state: ServerState = request.app["state"]
    data = await request.json()
    action = data.get("action")

    if action == "stop":
        state.stop_event.set()
        return web.json_response({"result": "stopping"})

    if action == "status":
        return await status_handler(request)

    return web.json_response(
        {"error": "unsupported action", "supported_actions": ["stop", "status"]},
        status=400,
    )


async def main(args):
    model = load_model(args.model)
    state = ServerState(
        model_path=args.model,
        recv_port=args.recv_port,
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        threshold=args.threshold,
        num_channels=args.num_channels,
    )

    data_queue = Queue()
    thread = threading.Thread(
        target=udp_stream_worker,
        args=(
            data_queue,
            state,
            args.recv_port,
            args.num_channels,
            args.band_names,
            args.timeout,
            args.max_windows,
        ),
        daemon=True,
    )
    thread.start()

    app = web.Application()
    app["state"] = state
    app.add_routes(
        [
            web.get("/ws", websocket_handler),
            web.get("/status", status_handler),
            web.get("/schema", schema_handler),
            web.post("/control", control_handler),
        ]
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.ws_host, args.ws_port)
    await site.start()
    logger.info("Server listening at http://%s:%d", args.ws_host, args.ws_port)
    logger.info("WebSocket endpoint available at ws://%s:%d/ws", args.ws_host, args.ws_port)

    await broadcast_loop(data_queue, model, args.threshold, state)
    await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket and HTTP server for live EEG focus classification")
    parser.add_argument("--model", required=True, help="Path to a saved sklearn model file")
    parser.add_argument("--recv-port", type=int, default=12345, help="UDP port for OpenBCI band-power input")
    parser.add_argument("--ws-host", default="0.0.0.0", help="Host address for the server")
    parser.add_argument("--ws-port", type=int, default=8765, help="Port for the server")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for focused classification")
    parser.add_argument("--num-channels", type=int, default=4, help="Number of channels expected per OpenBCI window")
    parser.add_argument("--timeout", type=float, default=None, help="UDP receive timeout in seconds, or None for indefinite listening")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum number of windows to process before exiting")
    parser.add_argument("--band-names", nargs="+", default=["delta", "theta", "alpha", "beta", "gamma"], help="Ordered band names from OpenBCI")
    args = parser.parse_args()

    asyncio.run(main(args))
