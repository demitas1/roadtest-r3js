import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

import { SceneProps, SceneObjects, GltfSceneData } from '../types/gltf'


const Scene = (sceneProps : SceneProps) => {
  const mountRef = useRef<HTMLDivElement>(null)
  // 立方体のマテリアルへの参照を保持するためのref
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null)
  const animationRef = useRef<number>(0)
  const color = sceneProps.testColor

  // refs to Three.js objects
  const scene: THREE.Scene | null = null;
  const model: THREE.Group | null = null;
  const materials: Map<string, THREE.Material> = new Map();
  const meshes: Map<string, THREE.Mesh> = new Map();


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
    camera.position.z = 5

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
    loadModel(scene, materials, meshes)

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

  // 色の変更を監視する新しいuseEffect
  useEffect(() => {
    if (materialRef.current && color) {
      // マテリアルが存在し、色情報が提供されている場合に色を更新
      materialRef.current.color.setRGB(color.r/255, color.g/255, color.b/255)
      materialRef.current.opacity = color.a/255
      materialRef.current.needsUpdate = true // マテリアルの更新を強制
    }
  }, [color]) // colorが変更されたときだけ実行

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
        'http://localhost:8000/static/TestCube.glb',
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

    // collect objects in the GLTF
    model.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        meshes.set(object.name, object)
        console.log('gltf: mesh')

        if (Array.isArray(object.material)) {
          object.material.forEach((mat, index) => {
            materials.set(`${object.name}_material_${index}`, mat)
            console.log(`${object.name}_material_${index}`)
          })
        } else {
          materials.set(`${object.name}_material`, object.material)
          console.log(`${object.name}_material`)
        }

        // シャドウの有効化
        object.castShadow = true
        object.receiveShadow = true
      }
    })

    // add to the scene
    scene.add(model)
  }

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}

export default Scene
