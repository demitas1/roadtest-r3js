import { useState, useEffect, useRef, useCallback } from 'react'
import Scene from './components/Scene'
import { MeshInfo } from './types/gltf'

function App() {
  const [message, setMessage] = useState<string>('Hello from React')
  const [responseSize, setResponseSize] = useState<number | null>(null)
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [statusMessage, setStatusMessage] = useState<string>('未接続')
  const websocketRef = useRef<WebSocket | null>(null)
  const [meshInfos, setMeshInfos] = useState<MeshInfo[]>([])
  const [selectedMesh, setSelectedMesh] = useState<string | null>(null)
  const [meshVisibility, setMeshVisibility] = useState<Record<string, boolean>>({})
  // メッシュの初期化済みフラグ
  const meshesInitializedRef = useRef<boolean>(false)

  // 立方体の色情報を保持するためのステート
  const [cubeColor, setCubeColor] = useState<{ r: number, g: number, b: number, a: number }>({
    r: 51, g: 102, b: 255, a: 255 // 初期色: #3366ff (不透明)
  })

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

          // バイナリデータを解析して色情報を取得
          const reader = new FileReader()

          reader.onload = () => {
            if (reader.result instanceof ArrayBuffer) {
              // ArrayBufferをUint8Arrayに変換
              const uint8Array = new Uint8Array(reader.result)

              // 最初の4バイトをRGBAとして解釈
              if (uint8Array.length >= 4) {
                const r = uint8Array[0]
                const g = uint8Array[1]
                const b = uint8Array[2]
                const a = uint8Array[3]

                console.log(`RGBA値: (${r}, ${g}, ${b}, ${a})`)

                // 色情報を更新
                setCubeColor({ r, g, b, a })
              }
            }
          }

          // BlobをArrayBufferとして読み込む
          reader.readAsArrayBuffer(event.data)
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

  // メッシュ情報が読み込まれたときのハンドラー
  const handleMeshesLoaded = useCallback((meshes: MeshInfo[]) => {
    console.log('メッシュ読み込み完了:', meshes.length);
    setMeshInfos(meshes);
    
    // 初回のみメッシュ選択と可視性を初期化
    if (!meshesInitializedRef.current && meshes.length > 0) {
      meshesInitializedRef.current = true;
      setSelectedMesh(meshes[0].name);
      
      // すべてのメッシュの可視性をデフォルトでtrueに設定
      const initialVisibility: Record<string, boolean> = {};
      meshes.forEach(mesh => {
        initialVisibility[mesh.name] = true;
      });
      setMeshVisibility(initialVisibility);
    }
  }, []);
  
  // メッシュの可視性を切り替える関数
  const toggleMeshVisibility = (meshName: string) => {
    setMeshVisibility(prev => ({
      ...prev,
      [meshName]: !prev[meshName]
    }));
  };

  return (
    <div className="app-container">
      <div className="scene-container">
        <Scene 
          testColor={cubeColor} 
          onMeshesLoaded={handleMeshesLoaded} 
          meshVisibility={meshVisibility}
        />
      </div>

      <div className="controls">
        <h2>WebSocket通信</h2>
        <p>状態: {statusMessage}</p>
        {responseSize !== null && (
          <p>最後に受信したデータのサイズ: {responseSize} バイト</p>
        )}
        <div className="color-display" style={{
          marginBottom: '15px',
          padding: '10px',
          backgroundColor: `rgba(${cubeColor.r}, ${cubeColor.g}, ${cubeColor.b}, ${cubeColor.a/255})`,
          color: ((cubeColor.r*0.299 + cubeColor.g*0.587 + cubeColor.b*0.114) > 186) ? '#000' : '#fff',
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          現在の色: RGB({cubeColor.r}, {cubeColor.g}, {cubeColor.b}, {cubeColor.a/255})
        </div>
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

        {/* メッシュ一覧セクション */}
        <div className="mesh-section">
          <h2>GLTFメッシュ一覧</h2>
          {meshInfos.length > 0 ? (
            <ul className="mesh-list">
              {meshInfos.map((mesh) => (
                <li 
                  key={mesh.name} 
                  className="mesh-item"
                  style={{ 
                    backgroundColor: selectedMesh === mesh.name ? '#444' : 'transparent',
                    padding: '8px',
                    marginBottom: '4px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center'
                  }}
                >
                  <label className="mesh-visibility-label">
                    <input 
                      type="checkbox" 
                      checked={meshVisibility[mesh.name] || false}
                      onChange={() => toggleMeshVisibility(mesh.name)}
                      onClick={(e) => e.stopPropagation()} // クリックイベントの伝播を停止
                    />
                  </label>
                  <div 
                    onClick={() => setSelectedMesh(mesh.name)}
                    style={{ flex: 1, marginLeft: '10px' }}
                  >
                    <div className="mesh-name">{mesh.name}</div>
                    <div className="mesh-position">
                      X: {mesh.position.x.toFixed(2)}, 
                      Y: {mesh.position.y.toFixed(2)}, 
                      Z: {mesh.position.z.toFixed(2)}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p>GLTFモデルの読み込み中...</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default App