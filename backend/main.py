from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn
import os
import random

from build_scene import example1


app = FastAPI()

# CORS設定を追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンのみを許可するように変更すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの提供
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    print("Warning: ./static directory doesn't exist. Static files won't be served.")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_binary(self, data: bytes, websocket: WebSocket):
        await websocket.send_bytes(data)

    async def send_json(self, data: dict, websocket: WebSocket):
        await websocket.send_json(data)

    async def send_text(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/")
async def get():
    return {"message": "WebSocket server is running"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # クライアントからメッセージを受信
            message = await websocket.receive_text()
            print(f"受信したメッセージ: {message}")

            # メッセージに応じて異なる処理を行う
            if message == "request json":
                # 「request json」を受信した場合、JSONデータを送信
                json_data = {"test message": "hello"}
                print(f"JSONデータを送信: {json_data}")
                await manager.send_json(json_data, websocket)
            elif message == "new scene1":
                # 「new scene1」を受信した場合、シーン1へ切り替え指示を送信
                json_data = {"new scene": "scene1"}
                print(f"シーン切り替えコマンドを送信: {json_data}")
                await manager.send_json(json_data, websocket)
            elif message == "new scene2":
                json_data = {"new scene": "scene2"}
                example1.create_example_scene()
                print(f"シーン切り替えコマンドを送信: {json_data}")
                await manager.send_json(json_data, websocket)
            else:
                # 通常の応答（ランダムなカラーを送信）
                # RGBA値をランダムに生成
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                a = random.randint(128, 255)  # 半透明〜不透明

                # バイナリデータとして送信
                binary_data = bytes([r, g, b, a])
                print(f"バイナリデータを送信: RGBA({r}, {g}, {b}, {a})")
                await manager.send_binary(binary_data, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("クライアントが切断しました")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
