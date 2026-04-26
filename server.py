import asyncio
import websockets
import json
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pythonosc import dispatcher, osc_server

# ==========================================
# 1. FastAPI 初始化与跨域设置 (HTTP 服务器)
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有来源请求
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局存放实时更新的脑波数据
current_data = {
    "relaxed": 0.0,
    "attention": 0.0,
    "engagement": 0.0
}

# ==========================================
# 2. OSC 接收器逻辑 (处理同学发来的数据)
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
# 3. HTTP 接口逻辑 (提供给前端的 /classify 和 /health)
# ==========================================
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Backend is running smoothly!"}

# 定义前端传过来的数据格式
class BrainData(BaseModel):
    theta: float
    alpha: float
    beta: float

@app.post("/classify")
def classify_data(data: BrainData):
    # 这里是你的 AI 分类逻辑，我先写一个简单的阈值判断作为占位
    # 根据 alpha (放松) 和 beta (专注) 的大小对比来返回状态
    if data.beta > data.alpha and data.beta > 0.5:
        return {"state": "focus"}
    elif data.alpha > data.beta and data.alpha > 0.5:
        return {"state": "relax"}
    else:
        return {"state": "idle"}

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
# 5. 启动统筹 (当 FastAPI 启动时，顺便启动 OSC 和 WebSocket)
# ==========================================
@app.on_event("startup")
async def startup_event():
    # 在后台线程启动 OSC (不会阻塞主程序)
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()
    
    # 在 asyncio 事件循环中启动 WebSocket 服务器
    asyncio.create_task(start_websocket_server())

# 程序入口
if __name__ == "__main__":
    print("🌟 正在启动服务器...")
    # 启动 8000 端口的 FastAPI 服务器
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)