import trimesh
import numpy as np
from PIL import Image


def example_scene():
    """
    trimeshを使ってシーングラフを構築する

    Returns:
        trimesh.Scene: 構築されたシーン
    """
    # Mesh 1 "triangle1"
    # ビットマップをテクスチャ用に読み込む
    texture_1_path = "./static/TestColorGrid.png"
    texture_1_img = Image.open(texture_1_path)

    # 頂点座標を設定
    vertices_1 = np.array([
        [0, 0, 0],  # 頂点0
        [0, 1, 0],  # 頂点1
        [1, 0, 0]   # 頂点2
    ], dtype=np.float32)

    # 面を設定（三角形1つ）
    faces_1 = np.array([
        [0, 2, 1]  # 頂点0, 1, 2を結ぶ三角形
    ])

    # UV座標を設定
    uvs_1 = np.array([
        [0, 0],  # 頂点0のUV: 左下
        [0, 1],  # 頂点1のUV: 左上
        [1, 0]   # 頂点2のUV: 右下
    ], dtype=np.float32)

    # trimeshでメッシュを生成
    visual_1 = trimesh.visual.TextureVisuals(uv=uvs_1, image=texture_1_img)
    triangle_1_mesh = trimesh.Trimesh(vertices=vertices_1, faces=faces_1, visual=visual_1, process=False)
    triangle_1_mesh.metadata['name'] = 'triangle1'

    # Mesh 2 "triangle2"
    # ビットマップをテクスチャ用に読み込む
    texture_2_path = "./static/TestPicture.png"
    texture_2_img = Image.open(texture_2_path)

    # 頂点座標を設定
    vertices_2 = np.array([
        [0, 0, 0],  # 頂点0
        [0, 1, 0],  # 頂点1
        [1, 0, 0]   # 頂点2
    ], dtype=np.float32)

    # 面を設定（三角形1つ）
    faces_2 = np.array([
        [0, 2, 1]  # 頂点0, 1, 2を結ぶ三角形
    ])

    # UV座標を設定
    uvs_2 = np.array([
        [0, 0],  # 頂点0のUV: 左下
        [0, 1],  # 頂点1のUV: 左上
        [1, 0]   # 頂点2のUV: 右下
    ], dtype=np.float32)

    # trimeshでメッシュを生成
    visual_2 = trimesh.visual.TextureVisuals(uv=uvs_2, image=texture_2_img)
    triangle_2_mesh = trimesh.Trimesh(vertices=vertices_2, faces=faces_2, visual=visual_2, process=False)
    triangle_2_mesh.metadata['name'] = 'triangle2'

    # 完全にクリーンな状態から始める
    scene = trimesh.Scene()

    # まず、各ジオメトリをシーンに追加（親子関係なし）
    world_geom = trimesh.Trimesh()
    world_geom.metadata['name'] = 'world'
    scene.add_geometry(world_geom, node_name='world')
    scene.add_geometry(triangle_1_mesh, node_name='triangle1')
    scene.add_geometry(triangle_2_mesh, node_name='triangle2')

    # 変換行列を準備
    identity = np.eye(4)
    triangle2_transform = np.eye(4)
    triangle2_transform[:3, 3] = [1.0, 1.0, 0.0]  # X,Y,Z方向の移動

    # 手動で親子関係と変換を設定
    # 親子関係を外部に保存
    custom_hierarchy = {
        'world': None,          # worldは親なし
        'triangle1': 'world',   # triangle1の親はworld
        'triangle2': 'triangle1' # triangle2の親はtriangle1
    }

    # 変換行列を外部に保存
    custom_transforms = {
        'world': identity,
        'triangle1': identity,
        'triangle2': triangle2_transform
    }

    print("\nManually defined hierarchy:")
    for node, parent in custom_hierarchy.items():
        print(f"Node: {node}, Parent: {parent}")

    print("\nManually defined transforms:")
    for node, transform in custom_transforms.items():
        if node == 'triangle2':
            print(f"Node: {node}, Transform:\n{transform}")

    # シーンとは別に、親子関係と変換を保存しておく
    # 後でconvert_to_glbに渡す
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


