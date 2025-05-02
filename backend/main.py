# main.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import random

app = FastAPI()

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では特定のオリジンに制限すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "FastAPI WebSocket Server"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # クライアントからのメッセージを待機
            data = await websocket.receive_text()
            print(f"受信したメッセージ: {data}")

            # データを処理し、バイナリレスポンスを生成
            binary_response = generate_binary_response(data)

            # バイナリデータをクライアントに送信
            await websocket.send_bytes(binary_response)

            # 次のメッセージを待つ前に少し待機
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"エラー発生: {str(e)}")
    finally:
        # 何らかの理由で接続が閉じられた場合に実行
        print("WebSocket接続終了")

def generate_binary_response(message: str) -> bytes:
    """
    受信したメッセージに基づいてダミーのバイナリデータを生成する

    この例では、メッセージの長さに基づいて異なるバイナリデータを生成
    """
    # メッセージの長さに基づいて簡単なバイナリデータを生成
    message_bytes = message.encode('utf-8')

    # ダミーのバイナリヘッダ (8バイト)
    header = bytes([0xAA, 0xBB, 0xCC, 0xDD, len(message_bytes) & 0xFF, (len(message_bytes) >> 8) & 0xFF, 0xEE, 0xFF])

    # ランダムなバイナリデータを生成 (0-255の値を持つランダムな10バイト)
    random_data = bytes([random.randint(0, 255) for _ in range(10)])

    # 元のメッセージとレスポンスを組み合わせる
    timestamp = bytes([random.randint(0, 255) for _ in range(4)])  # ダミーのタイムスタンプ

    # すべてを連結
    response = random_data

    return response

if __name__ == "__main__":
    # Uvicornサーバーを起動
    uvicorn.run(app, host="0.0.0.0", port=8000)
