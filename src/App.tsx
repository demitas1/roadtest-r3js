import { useState, useEffect, useRef } from 'react'
import Scene from './components/Scene'

function App() {
  const [message, setMessage] = useState<string>('Hello from React')
  const [responseSize, setResponseSize] = useState<number | null>(null)
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [statusMessage, setStatusMessage] = useState<string>('未接続')
  const websocketRef = useRef<WebSocket | null>(null)

  // WebSocketの初期化
  useEffect(() => {
    // WebSocketの接続
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws')

      ws.onopen = () => {
        console.log('WebSocket接続完了')
        setIsConnected(true)
        setStatusMessage('接続済み')
      }

      ws.onmessage = (event) => {
        // バイナリデータを受信
        if (event.data instanceof Blob) {
          // レスポンスサイズを表示
          setResponseSize(event.data.size)
          console.log(`受信したバイナリデータのサイズ: ${event.data.size} バイト`)

          // ここで必要に応じてバイナリデータを処理（今回は表示のみ）
        }
      }

      ws.onclose = () => {
        console.log('WebSocket接続が閉じられました')
        setIsConnected(false)
        setStatusMessage('接続が閉じられました')

        // 再接続を試みる（5秒後）
        setTimeout(() => {
          if (websocketRef.current === null || websocketRef.current.readyState === WebSocket.CLOSED) {
            connectWebSocket()
          }
        }, 5000)
      }

      ws.onerror = (error) => {
        console.error('WebSocketエラー:', error)
        setStatusMessage('接続エラー')
      }

      websocketRef.current = ws
    }

    connectWebSocket()

    // コンポーネントのクリーンアップ時にWebSocket接続を閉じる
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close()
      }
    }
  }, [])

  // バックエンドにメッセージを送信する関数
  const sendMessage = () => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      console.log(`メッセージを送信: ${message}`)
      websocketRef.current.send(message)
    } else {
      console.error('WebSocket接続が確立されていません')
      setStatusMessage('送信失敗：接続がありません')
    }
  }

  return (
    <div className="app-container">
      <div className="scene-container">
        <Scene />
      </div>

      <div className="controls">
        <h2>WebSocket通信</h2>
        <p>状態: {statusMessage}</p>
        {responseSize !== null && (
          <p>最後に受信したデータのサイズ: {responseSize} バイト</p>
        )}
        <div className="input-group">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
          />
          <button
            onClick={sendMessage}
            disabled={!isConnected}
          >
            送信
          </button>
        </div>
      </div>
    </div>
  )
}

export default App

