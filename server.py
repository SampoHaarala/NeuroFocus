import asyncio
import websockets
import json
from pythonosc import dispatcher
from pythonosc import osc_server
import threading

# 存放实时更新的脑波数据
current_data = {
    "relaxed": 0.0,
    "attention": 0.0,
    "engagement": 0.0
}

# --- 1. OSC 接收器逻辑 (处理同学发来的数据) ---
def relaxed_handler(address, value):
    current_data["relaxed"] = value

def attention_handler(address, value):
    current_data["attention"] = value

def engagement_handler(address, value):
    current_data["engagement"] = value

def start_osc_server():
    disp = dispatcher.Dispatcher()
    # 严格匹配你同学代码里的 OSC 路径
    disp.map("/focus/relaxed", relaxed_handler)
    disp.map("/focus/attention", attention_handler)
    disp.map("/focus/engagement", engagement_handler)
    
    # "0.0.0.0" 意味着接收局域网内发给这台电脑的所有数据
    # 端口 5005 必须和你同学的 OUT_PORT 对应
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 5005), disp)
    print("📡 OSC 接收器已启动，正在监听 5005 端口，等待同学的数据...")
    server.serve_forever()

# --- 2. WebSocket 服务器逻辑 (把数据喂给你的网页) ---
async def send_real_data(websocket):
    print("✅ 前端网页已连接！正在向网页推送真实数据...")
    try:
        while True:
            # 每 50 毫秒向网页发送一次最新的 current_data
            await websocket.send(json.dumps(current_data))
            await asyncio.sleep(0.05)
    except websockets.exceptions.ConnectionClosed:
        print("❌ 前端网页已断开连接。")

async def main_ws():
    async with websockets.serve(send_real_data, "localhost", 8080):
        print("🚀 WebSocket 服务器已启动 (ws://localhost:8080)")
        await asyncio.Future()

if __name__ == "__main__":
    # 启动 OSC 接收线程（后台静默运行）
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    # 启动 WebSocket 服务（主线程运行）
    asyncio.run(main_ws())