import { useRef, useEffect } from 'react'
import * as THREE from 'three'

const Scene = () => {
  const mountRef = useRef<HTMLDivElement>(null)
  
  useEffect(() => {
    if (!mountRef.current) return
    
    // まず、マウント要素に既にcanvasがないか確認
    // もし子要素があれば、すべて削除
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild)
    }
    
    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x222222)

    // コンテナの寸法を取得
    const containerWidth = mountRef.current.clientWidth
    const containerHeight = mountRef.current.clientHeight

    // Camera
    const camera = new THREE.PerspectiveCamera(
      75, // Field of view
      containerWidth / containerHeight, // Aspect ratio
      0.1, // Near clipping plane
      100 // Far clipping plane
    )
    camera.position.z = 5

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(containerWidth, containerHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    
    // Add canvas to the container
    mountRef.current.appendChild(renderer.domElement)

    // Create a cube
    const geometry = new THREE.BoxGeometry(2, 2, 2)
    const material = new THREE.MeshStandardMaterial({
      color: 0x3366ff,
      roughness: 0.4,
      metalness: 0.3
    })
    const cube = new THREE.Mesh(geometry, material)
    scene.add(cube)

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1)
    directionalLight.position.set(5, 5, 5)
    scene.add(directionalLight)

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate)
      
      // Rotate the cube
      cube.rotation.x += 0.01
      cube.rotation.y += 0.01
      
      renderer.render(scene, camera)
    }

    // Handle window resize
    const handleResize = () => {
      if (!mountRef.current) return
      
      const containerWidth = mountRef.current.clientWidth
      const containerHeight = mountRef.current.clientHeight

      camera.aspect = containerWidth / containerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(containerWidth, containerHeight)
    }

    window.addEventListener('resize', handleResize)
    
    // Start animation loop
    animate()

    // Clean up
    return () => {
      window.removeEventListener('resize', handleResize)
      
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

  return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
}

export default Scene
