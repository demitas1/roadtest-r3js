import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

import { SceneProps, SceneObjects, GltfSceneData, MeshInfo } from '../types/gltf'

const Scene = (sceneProps: SceneProps) => {
  const mountRef = useRef<HTMLDivElement>(null)
  // 立方体のマテリアルへの参照を保持するためのref
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null)
  const animationRef = useRef<number>(0)
  const color = sceneProps.testColor
  const meshVisibility = sceneProps.meshVisibility || {}
  
  // Three.js オブジェクトへの参照をrefに保存
  const meshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const materialsRef = useRef<Map<string, THREE.Material>>(new Map());

  // メッシュ情報を収集
  const [meshInfos, setMeshInfos] = useState<MeshInfo[]>([])

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

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x222222)

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75, // Field of view
      containerWidth / containerHeight, // Aspect ratio
      0.1, // Near clipping plane
      100 // Far clipping plane
    )
    camera.position.z = 20

    // Create a cube
    const geometry = new THREE.BoxGeometry(2, 2, 2)
    const material = new THREE.MeshStandardMaterial({
      color: color ? new THREE.Color(color.r/255, color.g/255, color.b/255) : 0x3366ff,
      roughness: 0.4,
      metalness: 0.3,
      transparent: true,
      opacity: color ? color.a/255 : 1.0
    })
    materialRef.current = material  // マテリアルへの参照を保存
    const cube = new THREE.Mesh(geometry, material)
    scene.add(cube)

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(5, 5, 5)
    scene.add(directionalLight)

    // load the scene from GLTF
    loadModel(scene, materialsRef.current, meshesRef.current)

    // Handle window resize
    const handleResize = () => {
      if (!mountRef.current) return

      const containerWidth = mountRef.current.clientWidth
      const containerHeight = mountRef.current.clientHeight

      camera.aspect = containerWidth / containerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(containerWidth, containerHeight)
    }
    // add resize event listener
    window.addEventListener('resize', handleResize)

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate)

      // Rotate the cube
      cube.rotation.x += 0.01
      cube.rotation.y += 0.01

      // TEST: access to gltf individual objects
      const cube_1 = meshesRef.current.get('Cube')
      if (cube_1) {
        cube_1.rotation.x += 0.01
      }

      const icosphere_1 = meshesRef.current.get('Icosphere')
      if (icosphere_1) {
        icosphere_1.rotation.y += 0.01
      }

      renderer.render(scene, camera)
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

      // unmount DOM element
      if (mountRef.current) {
        mountRef.current.removeChild(renderer.domElement)
      }

      // Dispose of Three.js resources to prevent memory leaks
      geometry.dispose()
      material.dispose()
      scene.remove(cube)
      renderer.dispose()
    }
  }, []) // Empty dependency array means this effect runs once on mount

  // メッシュ情報が更新されたときに親コンポーネントに通知
  useEffect(() => {
    if (meshInfos.length > 0 && sceneProps.onMeshesLoaded) {
      sceneProps.onMeshesLoaded(meshInfos);
    }
  }, [meshInfos, sceneProps.onMeshesLoaded]);

  // 色の変更を監視する新しいuseEffect
  useEffect(() => {
    if (materialRef.current && color) {
      // マテリアルが存在し、色情報が提供されている場合に色を更新
      materialRef.current.color.setRGB(color.r/255, color.g/255, color.b/255)
      materialRef.current.opacity = color.a/255
      materialRef.current.needsUpdate = true // マテリアルの更新を強制
    }
  }, [color]) // colorが変更されたときだけ実行
  
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

  //
  // load objects to the scene
  //

  const loadModel = (
    scene: THREE.Scene,
    materials: Map<string, THREE.Material>,
    meshes: Map<string, THREE.Mesh>
  ) => {
    const loader = new GLTFLoader()

    // 型によって読み込み方法を分岐
    if (true) {
      // URLからロード
      loader.load(
        'http://localhost:8000/static/TestScene.glb',
        (gltf) => onModelLoaded(gltf, scene, materials, meshes),
        (xhr) => {
          console.log((xhr.loaded / xhr.total * 100) + '% loaded')
        },
        (error) => {
          console.error('GLTFLoader error:', error)
        }
      )
    } else {
      // JSONオブジェクトから直接ロード
      loader.parse(
        JSON.stringify(gltfData.source),
        '',
        (gltf) => onModelLoaded(gltf, scene, materials, meshes),
        (error) => {
          console.error('GLTFLoader parsing error:', error)
        }
      )
    }
  }

  const onModelLoaded = (
    gltf: GLTF,
    scene: THREE.Scene,
    materials: Map<string, THREE.Material>,
    meshes: Map<string, THREE.Mesh>
  ) => {
    const model = gltf.scene
    console.log('GLTF load complete.')

    const loadedMeshInfos: MeshInfo[] = [];

    // collect objects in the GLTF
    model.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        meshes.set(object.name, object)
        console.log(`gltf: mesh: ${object.name}`)

        let materialName = '';
        if (Array.isArray(object.material)) {
          object.material.forEach((mat, index) => {
            const matName = `${object.name}_material_${index}`;
            materials.set(matName, mat)
            console.log(`material: ${matName}`)
            if (index === 0) materialName = matName;
          })
        } else {
          const matName = `${object.name}_material`;
          materials.set(matName, object.material)
          console.log(`material: ${matName}`)
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

    // メッシュ情報を設定
    setMeshInfos(loadedMeshInfos);

    // add to the scene
    scene.add(model)
    
    // モデルが読み込まれた後で、現在のvisibility設定を適用
    // これは特に重要で、最初のレンダリング時にvisibility設定を反映するために必要
    if (Object.keys(meshVisibility).length > 0) {
      console.log("初期visibilityを適用します:", meshVisibility);
      Object.entries(meshVisibility).forEach(([meshName, isVisible]) => {
        const mesh = meshes.get(meshName);
        if (mesh) {
          console.log(`初期状態: メッシュ「${meshName}」のvisibilityを${isVisible ? '表示' : '非表示'}に設定`);
          mesh.visible = isVisible;
        }
      });
    }
  }

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}

export default Scene