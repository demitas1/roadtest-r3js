import * as THREE from 'three'

// glTFデータの型定義
export interface GltfSceneData {
  // ファイルURLまたはglTFのJSON構造
  source: string | object | null;

  // シーン設定
  settings?: {
    backgroundColor?: string;
    ambientLight?: {
      color?: string;
      intensity?: number;
    };
    directionalLight?: {
      color?: string;
      intensity?: number;
      position?: [number, number, number];
    };
    camera?: {
      position?: [number, number, number];
      lookAt?: [number, number, number];
      fov?: number;
      near?: number;
      far?: number;
    };
    useOrbitControls?: boolean;
  };

  // オブジェクト変換情報のオーバーライド
  overrides?: {
    [objectName: string]: {
      visible?: boolean;
      position?: [number, number, number];
      rotation?: [number, number, number]; // Euler回転 (ラジアン)
      scale?: [number, number, number];
      material?: {
        color?: string;
        opacity?: number;
        roughness?: number;
        metalness?: number;
        emissive?: string;
        emissiveIntensity?: number;
      };
    };
  };
}

// メッシュ情報の型定義
export interface MeshInfo {
  name: string;
  materialName: string;
  position: THREE.Vector3;
  rotation: THREE.Euler;
  scale: THREE.Vector3;
  vertexCount: number;
  triangleCount: number;
}

// シーンコンポーネントのProps
export interface SceneProps {
  gltfData?: GltfSceneData;
  testColor?: {
    r: number;
    g: number;
    b: number;
    a: number;
  }
  onMeshesLoaded?: (meshInfos: MeshInfo[]) => void;
  meshVisibility?: Record<string, boolean>;
  modelUrl?: string;  // 新しいモデルURLプロパティ
  reloadTrigger?: number;  // シーン再読み込みトリガープロパティ
}

// シーンオブジェクトの型定義
export interface SceneObjects {
  scene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  renderer: THREE.WebGLRenderer | null;
  controls: any | null; // OrbitControlsの型をインポートしない場合
  model: THREE.Group | null;
  materials: Map<string, THREE.Material>;
  meshes: Map<string, THREE.Mesh>;
}