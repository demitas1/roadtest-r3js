import trimesh
import numpy as np
from PIL import Image


def empty_scene():
    """
    空のシーングラフを構築する（worldノードのみ）

    Returns:
        trimesh.Scene: 構築されたシーン（worldノードのみ）
    """
    # 完全にクリーンな状態から始める
    scene = trimesh.Scene()

    # worldジオメトリをシーンに追加
    # 注意: 完全に空のメッシュではなく、最小限の頂点と面を持つメッシュを作成
    # GLTFビューワーが空のメッシュを扱えない可能性があるため
    vertices = np.array([[0, 0, 0]], dtype=np.float32)  # 1つの頂点（原点）
    world_geom = trimesh.Trimesh(vertices=vertices, process=False)
    world_geom.metadata['name'] = 'world'
    scene.add_geometry(world_geom, node_name='world')

    # 変換行列を準備
    identity = np.eye(4)

    # 親子関係と変換を初期化
    custom_hierarchy = {
        'world': None  # worldは親なし
    }

    custom_transforms = {
        'world': identity
    }

    # シーンのメタデータに保存
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


def add_mesh_triangle(
        scene,
        name,
        vertices,
        faces,
        uvs,
        texture_path,
        position=None,
        parent_node=None):
    """
    三角形のメッシュをシーンに追加する

    Args:
        scene (trimesh.Scene): メッシュを追加するシーン
        name (str): メッシュの名前
        texture_path (str): テクスチャ画像のパス
        position (list, optional): [x, y, z]の位置。Noneの場合は[0, 0, 0]
        parent_node (str, optional): 親ノードの名前。Noneの場合は'world'

    Returns:
        trimesh.Scene: 更新されたシーン
    """
    # デフォルト値の設定
    if position is None:
        position = [0, 0, 0]

    if parent_node is None:
        parent_node = 'world'

    # ビットマップをテクスチャ用に読み込む
    texture_img = Image.open(texture_path)

    # TODO: 検査
    # 頂点座標を設定
    # 面を設定（三角形1つ）
    # UV座標を設定

    # trimeshでメッシュを生成
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_img)
    triangle_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    triangle_mesh.metadata['name'] = name

    # シーンに追加
    scene.add_geometry(triangle_mesh, node_name=name)

    # 変換行列を準備
    transform = np.eye(4)
    transform[:3, 3] = position  # X,Y,Z方向の移動

    # 既存のメタデータを取得
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    # 親子関係と変換を更新
    custom_hierarchy[name] = parent_node
    custom_transforms[name] = transform

    # シーンのメタデータに保存し直す
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


def example_scene():
    """
    trimeshを使ってシーングラフを構築する（リファクタリング後のバージョン）

    Returns:
        trimesh.Scene: 構築されたシーン
    """
    # 空のシーンを作成
    scene = empty_scene()

    # triangle1を追加（親はworld）
    scene = add_mesh_triangle(
        scene=scene,
        name='triangle1',
        # trimesh は +Y up
        vertices = np.array([
            [0, 0, 0],  # 頂点0
            [0, 1, 0],  # 頂点1
            [1, 0, 0],  # 頂点2
            [1, 1, 0],  # 頂点3
        ], dtype=np.float32),
        faces = np.array([
            [0, 2, 1],
            [1, 2, 3],
        ]),
        # trimesh では v=0 が画像の上端となる?
        uvs = np.array([
            [0, 0],  # 頂点0のUV: 左下
            [0, 1],  # 頂点1のUV: 左上
            [1, 0],  # 頂点2のUV: 右下
            [1, 1],  # 頂点2のUV: 右上
        ], dtype=np.float32),
        texture_path='./static/TestColorGrid.png',
        position=[0, 0, 0],
        parent_node='world'
    )

    # triangle2を追加（親はtriangle1）
    scene = add_mesh_triangle(
        scene=scene,
        name='triangle2',
        vertices = np.array([
            [0, 0, 0],  # 頂点0
            [0, 1, 0],  # 頂点1
            [1, 0, 0],  # 頂点2
            [1, 1, 0],  # 頂点3
        ], dtype=np.float32),
        faces = np.array([
            [0, 2, 1],
            [1, 2, 3],
        ]),
        uvs = np.array([
            [0, 0],  # 頂点0のUV: 左下
            [0, 1],  # 頂点1のUV: 左上
            [1, 0],  # 頂点2のUV: 右下
            [1, 1],  # 頂点2のUV: 右上
        ], dtype=np.float32),
        texture_path='./static/TestPicture.png',
        position=[1.0, 1.0, 0.0],
        parent_node='triangle1'
    )

    # デバッグ出力
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    print("\nManually defined hierarchy:")
    for node, parent in custom_hierarchy.items():
        print(f"Node: {node}, Parent: {parent}")

    print("\nManually defined transforms:")
    for node, transform in custom_transforms.items():
        if node == 'triangle2':
            print(f"Node: {node}, Transform:\n{transform}")

    return scene
