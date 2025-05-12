import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

import { SceneProps, SceneObjects, GltfSceneData, MeshInfo } from '../types/gltf'

const Scene = (sceneProps: SceneProps) => {
  const mountRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number>(0)
  const meshVisibility = sceneProps.meshVisibility || {}
  const modelUrl = sceneProps.modelUrl || ''
  const reloadTrigger = sceneProps.reloadTrigger || 0
  
  // Three.js オブジェクトへの参照を保存
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const lightsRef = useRef<THREE.Light[]>([])
  const modelRef = useRef<THREE.Group | null>(null)
  
  // Three.js オブジェクトへの参照をrefに保存
  const meshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const materialsRef = useRef<Map<string, THREE.Material>>(new Map());
  // テクスチャを追跡するための新しいMap
  const texturesRef = useRef<Map<string, THREE.Texture>>(new Map());

  // メッシュ情報を収集
  const [meshInfos, setMeshInfos] = useState<MeshInfo[]>([])

  // シーンの初期化
  useEffect(() => {
    if (!mountRef.current) return

    // まず、マウント要素に既にcanvasがないか確認
    // もし子要素があれば、すべて削除
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild)
    }

    // get container size
    const containerWidth = mountRef.current.clientWidth
    const containerHeight = mountRef.current.clientHeight

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(containerWidth, containerHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    // Add canvas to the container
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x222222)
    sceneRef.current = scene

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75, // Field of view
      containerWidth / containerHeight, // Aspect ratio
      0.1, // Near clipping plane
      100 // Far clipping plane
    )
    camera.position.z = 20
    cameraRef.current = camera

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.25
    controlsRef.current = controls

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    sceneRef.current.add(ambientLight)
    lightsRef.current.push(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(5, 5, 5)
    sceneRef.current.add(directionalLight)
    lightsRef.current.push(directionalLight)

    // load the scene from GLTF
    loadModel()

    // Handle window resize
    const handleResize = () => {
      if (!mountRef.current || !cameraRef.current || !rendererRef.current) return

      const containerWidth = mountRef.current.clientWidth
      const containerHeight = mountRef.current.clientHeight

      cameraRef.current.aspect = containerWidth / containerHeight
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(containerWidth, containerHeight)
    }
    // add resize event listener
    window.addEventListener('resize', handleResize)

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate)

      // Rotate the objects if it exists
      // TEST: access to gltf individual objects
      const cube_1 = meshesRef.current.get('Cube')
      if (cube_1) {
        cube_1.rotation.x += 0.01
      }

      const icosphere_1 = meshesRef.current.get('Icosphere')
      if (icosphere_1) {
        icosphere_1.rotation.y += 0.01
      }

      // Update controls
      if (controlsRef.current) {
        controlsRef.current.update()
      }

      // Render scene
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }
    }
    // Start animation loop
    animate()

    // Clean up function
    return () => {
      // remove event listener
      window.removeEventListener('resize', handleResize)

      // delete animation callback
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }

      // 最終的なクリーンアップとしてモデルを処理
      clearCurrentModel();

      // Clean up renderer
      if (rendererRef.current && mountRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement)
        rendererRef.current.dispose()
      }
    }
  }, []) // Empty dependency array means this effect runs once on mount


  // ファイルの存在確認やURLのチェックを行う関数
  const checkFileExists = async (url: string): Promise<boolean> => {
    try {
      // HEAD リクエストを送信して、ファイルの存在のみを確認
      console.log(`checking url: ${url}`)
      const response = await fetch(url, { method: 'HEAD' });
      return response.ok;
    } catch (error) {
      console.error('File existence check failed:', error);
      return false;
    }
  }

  // モデルのロード関数
  const loadModel = async (clearAllSceneObjects = false) => {
    if (!sceneRef.current) return

    // 既存のモデルがあれば削除
    // 引数で指定されたオプションを渡して、シーン内のすべてのオブジェクトをクリアするか決定
    clearCurrentModel(clearAllSceneObjects)
    
    let url = modelUrl;
    if (url.length === 0) {
      return;
    }

    // TODO:
    // try...catchを使って書き直す
    // checkFileExsits と GLTFLoader のエラーを一括して catch できるように
    // GLTFLoader も promise でラップする
    if (await checkFileExists(modelUrl)) {
      // URLからモデルをロード
      const loader = new GLTFLoader()
      console.log(`モデルをロード中: ${url}`)
      loader.load(
        url,
        (gltf) => onModelLoaded(gltf),
        (xhr) => {
          console.log((xhr.loaded / xhr.total * 100) + '% loaded')
        },
        (error) => {
          console.error('GLTFLoader error:', error)
        }
      )
    } else {
      // not found error
      console.error(`not found: ${url}`)
    }
  }
  
  // 現在のモデルをクリアする関数（拡張版）
  const clearCurrentModel = (clearAllSceneObjects = false) => {
    if (!sceneRef.current) return
    
    console.log('モデルクリア処理を開始します。' + (clearAllSceneObjects ? ' シーン内のすべてのオブジェクトをクリアします。' : ''));
    
    // 既存のモデルがあれば削除
    if (modelRef.current) {
      console.log(`モデルオブジェクトを削除します: ${modelRef.current.name || 'unnamed model'}`);
      
      // モデル内のすべてのリソースを適切に開放するためのトラバース
      modelRef.current.traverse((object) => {
        // オブジェクトタイプごとの適切な処理
        if (object instanceof THREE.Mesh) {
          console.log(`メッシュを処理: ${object.name}`);
          
          // ジオメトリを処理
          if (object.geometry) {
            console.log(`ジオメトリを処理: ${object.geometry.uuid}`);
            object.geometry.dispose();
          }
          
          // マテリアルを処理（配列またはシングル）
          if (Array.isArray(object.material)) {
            object.material.forEach((material, index) => {
              disposeMaterial(material, `${object.name}_material_${index}`);
            });
          } else if (object.material) {
            disposeMaterial(object.material, `${object.name}_material`);
          }
        }
        
        // 他の特殊なオブジェクトタイプのクリーンアップ処理をここに追加可能
      });
      
      // シーンからモデルを削除
      sceneRef.current.remove(modelRef.current);
      modelRef.current = null;
    }
    
    // オプション: シーン内のすべてのメッシュとマテリアルをクリア
    if (clearAllSceneObjects && sceneRef.current) {
      console.log('シーン内のすべてのメッシュとマテリアルをクリアします...');
      
      // 削除するオブジェクトを一時配列に保存（シーン構造を変更しながらのトラバースを避けるため）
      const objectsToRemove: THREE.Object3D[] = [];
      
      // シーン全体をトラバースして、メッシュとマテリアルを探す
      sceneRef.current.traverse((object) => {
        // メッシュの処理
        if (object instanceof THREE.Mesh) {
          console.log(`シーンから削除するメッシュを見つけました: ${object.name || 'unnamed mesh'}`);
          
          // ジオメトリの処理
          if (object.geometry) {
            console.log(`ジオメトリを処理: ${object.geometry.uuid}`);
            object.geometry.dispose();
          }
          
          // マテリアルの処理
          if (Array.isArray(object.material)) {
            object.material.forEach((material, index) => {
              disposeMaterial(material, `${object.name || 'unnamed'}_material_${index}`);
            });
          } else if (object.material) {
            disposeMaterial(object.material, `${object.name || 'unnamed'}_material`);
          }
          
          // 親が存在し、シーンそのものでない場合は削除対象に追加
          if (object.parent && object.parent !== sceneRef.current) {
            objectsToRemove.push(object);
          }
        }
        
        // 他の特定のオブジェクトタイプの処理も追加可能
        // 例: ライト、カメラなど
      });
      
      // 収集したオブジェクトをシーンから削除
      objectsToRemove.forEach(object => {
        if (object.parent) {
          console.log(`オブジェクトを親から削除: ${object.name || 'unnamed object'}`);
          object.parent.remove(object);
        }
      });
      
      console.log(`シーン内のオブジェクト削除完了。(${objectsToRemove.length}個削除)`);
    }
    
    // メッシュとマテリアルのマップを明示的にクリア
    console.log(`メッシュマップをクリア（${meshesRef.current.size}個のエントリ）`);
    meshesRef.current.clear();
    
    console.log(`マテリアルマップをクリア（${materialsRef.current.size}個のエントリ）`);
    materialsRef.current.clear();
    
    console.log(`テクスチャマップをクリア（${texturesRef.current.size}個のエントリ）`);
    texturesRef.current.clear();
    
    // メッシュ情報をクリア
    setMeshInfos([]);
    
    console.log('モデルクリア処理が完了しました。');
  }
  
  // マテリアルとそれに関連するリソースを適切に開放するヘルパー関数
  const disposeMaterial = (material: THREE.Material, name: string) => {
    console.log(`マテリアルを処理: ${name} (${material.uuid})`);
    
    // マテリアルに関連するすべてのテクスチャを処理
    if (material instanceof THREE.MeshStandardMaterial) {
      disposeTextureIfExists(material.map, `${name}_map`);
      disposeTextureIfExists(material.normalMap, `${name}_normalMap`);
      disposeTextureIfExists(material.roughnessMap, `${name}_roughnessMap`);
      disposeTextureIfExists(material.metalnessMap, `${name}_metalnessMap`);
      disposeTextureIfExists(material.aoMap, `${name}_aoMap`);
      disposeTextureIfExists(material.emissiveMap, `${name}_emissiveMap`);
    } else if (material instanceof THREE.MeshBasicMaterial) {
      disposeTextureIfExists(material.map, `${name}_map`);
    }
    // 他のマテリアルタイプのハンドリングもここに追加可能
    
    // マテリアル自体を破棄
    material.dispose();
  }
  
  // テクスチャが存在する場合にそれを適切に開放するヘルパー関数
  const disposeTextureIfExists = (texture: THREE.Texture | null, name: string) => {
    if (texture) {
      console.log(`テクスチャを処理: ${name} (${texture.uuid})`);
      texture.dispose();
    }
  }

  // モデルがロードされた時の処理
  const onModelLoaded = (gltf: GLTF) => {
    if (!sceneRef.current) return
    
    const model = gltf.scene
    console.log('GLTF load complete.')
    
    // モデルの参照を保存
    console.log(`gltf.scene: isGroup? = ${model.isGroup}`)
    modelRef.current = model

    const loadedMeshInfos: MeshInfo[] = [];

    // collect objects in the GLTF
    model.traverse((object) => {
      if (object.name) {
        console.log(`gltf: object: ${object.name}`)
      }
      if (object instanceof THREE.Mesh) {
        meshesRef.current.set(object.name, object)
        console.log(`gltf: mesh: ${object.name}`)

        let materialName = '';
        if (Array.isArray(object.material)) {
          object.material.forEach((mat, index) => {
            const matName = `${object.name}_material_${index}`;
            materialsRef.current.set(matName, mat)
            console.log(`material: ${matName}`)
            
            // マテリアルに関連するテクスチャを追跡
            trackMaterialTextures(mat, matName);
            
            if (index === 0) materialName = matName;
          })
        } else {
          const matName = `${object.name}_material`;
          materialsRef.current.set(matName, object.material)
          console.log(`material: ${matName}`)
          
          // マテリアルに関連するテクスチャを追跡
          trackMaterialTextures(object.material, matName);
          
          materialName = matName;
        }

        // シャドウの有効化
        object.castShadow = true
        object.receiveShadow = true

        // メッシュ情報を収集
        let vertexCount = 0;
        let triangleCount = 0;
        
        if (object.geometry) {
          const position = object.geometry.getAttribute('position');
          if (position) {
            vertexCount = position.count;
            triangleCount = Math.floor(vertexCount / 3);
          }
        }

        loadedMeshInfos.push({
          name: object.name,
          materialName,
          position: object.position.clone(),
          rotation: object.rotation.clone(),
          scale: object.scale.clone(),
          vertexCount,
          triangleCount
        });
      }
    })
    
    // マテリアルに関連するテクスチャを追跡するヘルパー関数
    function trackMaterialTextures(material: THREE.Material, matName: string) {
      if (material instanceof THREE.MeshStandardMaterial) {
        trackTextureIfExists(material.map, `${matName}_map`);
        trackTextureIfExists(material.normalMap, `${matName}_normalMap`);
        trackTextureIfExists(material.roughnessMap, `${matName}_roughnessMap`);
        trackTextureIfExists(material.metalnessMap, `${matName}_metalnessMap`);
        trackTextureIfExists(material.aoMap, `${matName}_aoMap`);
        trackTextureIfExists(material.emissiveMap, `${matName}_emissiveMap`);
      } else if (material instanceof THREE.MeshBasicMaterial) {
        trackTextureIfExists(material.map, `${matName}_map`);
      }
      // 他のマテリアルタイプもここに追加可能
    }
    
    // テクスチャを追跡するヘルパー関数
    function trackTextureIfExists(texture: THREE.Texture | null, name: string) {
      if (texture) {
        texturesRef.current.set(name, texture);
        console.log(`テクスチャを追跡: ${name} (${texture.uuid})`);
      }
    }

    // メッシュ情報を設定
    setMeshInfos(loadedMeshInfos);

    // add to the scene
    sceneRef.current.add(model)
    
    // モデルが読み込まれた後で、現在のvisibility設定を適用
    // これは特に重要で、最初のレンダリング時にvisibility設定を反映するために必要
    if (Object.keys(meshVisibility).length > 0) {
      console.log("初期visibilityを適用します:", meshVisibility);
      Object.entries(meshVisibility).forEach(([meshName, isVisible]) => {
        const mesh = meshesRef.current.get(meshName);
        if (mesh) {
          console.log(`初期状態: メッシュ「${meshName}」のvisibilityを${isVisible ? '表示' : '非表示'}に設定`);
          mesh.visible = isVisible;
        }
      });
    }
  }

  // メッシュ情報が更新されたときに親コンポーネントに通知
  useEffect(() => {
    if (meshInfos.length > 0 && sceneProps.onMeshesLoaded) {
      sceneProps.onMeshesLoaded(meshInfos);
    }
  }, [meshInfos, sceneProps.onMeshesLoaded]);

  // メッシュのvisibilityを監視するuseEffect
  useEffect(() => {
    console.log("changed visibility:", meshVisibility);
    // meshesRef.currentが空でないことを確認
    if (meshesRef.current.size > 0) {
      // meshVisibilityが更新されたら、対応するメッシュのvisibilityを変更
      Object.entries(meshVisibility).forEach(([meshName, isVisible]) => {
        const mesh = meshesRef.current.get(meshName);
        if (mesh) {
          console.log(`mesh "${meshName}" is set to ${isVisible ? 'visible' : 'not visible'}`);
          mesh.visible = isVisible;
        } else {
          console.log(`mesh "${meshName}" is not found`);
        }
      });
    } else {
      console.log("meshesRef is empty - no meshes are loaded");
    }
  }, [meshVisibility]); // meshesRef.currentは参照が変わらないのでここでは依存に含めない
  
  // モデルURLまたはリロードトリガーが変更されたときにシーンを再読み込み
  useEffect(() => {
    // シーン再読み込み
    if (sceneRef.current) {
      console.log(`シーンを再読み込みします。URL: ${modelUrl}, トリガー: ${reloadTrigger}`);
      // モデルURLが変わる際は、シーン内のすべてのオブジェクトをクリアする
      // これにより、モデル切り替え時の古いオブジェクト残留問題を解決
      const shouldClearAllObjects = true;
      loadModel(shouldClearAllObjects);
    }
  }, [modelUrl, reloadTrigger]);

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}

export default Scene
