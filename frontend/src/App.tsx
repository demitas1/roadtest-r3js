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
  // 現在ロードされているシーンのURL
  const [currentSceneUrl, setCurrentSceneUrl] = useState<string>('')
  // シーン再読み込みトリガー
  const [reloadScene, setReloadScene] = useState<number>(0)

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
                // do something

              }
            }
          }

          // BlobをArrayBufferとして読み込む
          reader.readAsArrayBuffer(event.data)
        } 
        // JSONデータを受信した場合の処理
        else if (typeof event.data === 'string') {
          try {
            const jsonData = JSON.parse(event.data)
            console.log('受信したJSONデータ:', jsonData)
            
            // JSONのサイズを表示
            setResponseSize(event.data.length)
            console.log(`受信したJSONデータのサイズ: ${event.data.length} バイト`)
            
            // シーン変更コマンドの処理
            if (jsonData['new scene'] === 'scene1') {
              console.log('シーン1への切り替えコマンドを受信しました')
              // 新しいシーンURLを設定
              setCurrentSceneUrl('http://localhost:8000/static/TestCube.glb')
              // メッシュ情報をリセット
              setMeshInfos([])
              setSelectedMesh(null)
              setMeshVisibility({})
              meshesInitializedRef.current = false
              // シーンの再読み込みをトリガー
              setReloadScene(prev => prev + 1)
              setStatusMessage('シーン1に切り替えました')
            } else if (jsonData['new scene'] === 'scene2' || jsonData['new scene'] === 'scene3') {
              console.log(`コマンドを受信しました ${jsonData}`)
              if ("gltf_path" in jsonData) {
                // 新しいシーンURLを設定
                const glbPath: string = jsonData["gltf_path"];
                const sceneUrl: string = `http://localhost:8000/${glbPath}`;
                console.log(`new scene url: ${sceneUrl}`);
                const scene_url = `http://localhost:8000/${glbPath}`
                setCurrentSceneUrl(scene_url)
                // メッシュ情報をリセット
                setMeshInfos([])
                setSelectedMesh(null)
                setMeshVisibility({})
                meshesInitializedRef.current = false
                // シーンの再読み込みをトリガー
                setReloadScene(prev => prev + 1)
                setStatusMessage('シーン2に切り替えました')
              }
            }
            // 色情報の処理
            else if (jsonData.color) {
              const { r, g, b, a = 255 } = jsonData.color
              if (r !== undefined && g !== undefined && b !== undefined) {
                setCubeColor({ r, g, b, a })
                console.log(`JSONから色情報を更新: RGB(${r}, ${g}, ${b}, ${a})`)
              }
            }
            // その他のテストメッセージ
            else if (jsonData['test message']) {
              console.log(`テストメッセージを受信: ${jsonData['test message']}`)
            }
          } catch (error) {
            console.log('受信したデータはJSONではありません:', event.data)
            // 非JSONテキストデータのサイズを表示
            setResponseSize(event.data.length)
          }
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
          onMeshesLoaded={handleMeshesLoaded} 
          meshVisibility={meshVisibility}
          modelUrl={currentSceneUrl}
          reloadTrigger={reloadScene}
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
          borderRadius: '4px',
          textAlign: 'center'
        }}>
          test message
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

        <div className="scene-info">
          <h3>現在のシーン</h3>
          <p>{currentSceneUrl.split('/').pop()}</p>
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
