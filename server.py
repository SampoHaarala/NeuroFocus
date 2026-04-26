import asyncio
import websockets
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pythonosc import dispatcher, osc_server

# ==========================================
# 1. FastAPI 初始化与跨域设置 (HTTP 服务器)
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_data = {
    "relaxed": 0.0,
    "attention": 0.0,
    "engagement": 0.0
}

# ==========================================
# 2. OSC 接收器逻辑 (处理脑电设备端发来的数据)
# ==========================================
def relaxed_handler(address, value):
    current_data["relaxed"] = value

def attention_handler(address, value):
    current_data["attention"] = value

def engagement_handler(address, value):
    current_data["engagement"] = value

def start_osc_server():
    disp = dispatcher.Dispatcher()
    disp.map("/focus/relaxed", relaxed_handler)
    disp.map("/focus/attention", attention_handler)
    disp.map("/focus/engagement", engagement_handler)
    
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 5005), disp)
    print("📡 [OSC] 接收器已启动，正在监听 5005 端口...")
    server.serve_forever()

# ==========================================
# 3. HTTP 接口逻辑 (仅作状态诊断用)
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is pure and running fast!"}

# ==========================================
# 4. WebSocket 服务器逻辑 (8080 端口实时推送)
# ==========================================
async def send_real_data(websocket):
    print("✅ [WebSocket] 前端网页已连接！正在推送数据...")
    try:
        while True:
            await websocket.send(json.dumps(current_data))
            await asyncio.sleep(0.05) # 50 毫秒推送一次
    except websockets.exceptions.ConnectionClosed:
        print("❌ [WebSocket] 前端网页已断开连接。")

async def start_websocket_server():
    async with websockets.serve(send_real_data, "localhost", 8080):
        print("🚀 [WebSocket] 服务器已启动 (ws://localhost:8080)")
        await asyncio.Future()

# ==========================================
# 5. 启动统筹
# ==========================================
@app.on_event("startup")
async def startup_event():
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()
    asyncio.create_task(start_websocket_server())

if __name__ == "__main__":
    print("🌟 正在启动服务器...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)