import trimesh
import numpy as np
from PIL import Image
import uuid


def empty_scene():
    """
    Empty scene

    Returns:
        trimesh.Scene: empty trimesh Scene object
    """
    scene = trimesh.Scene()

    # worldジオメトリをシーンに追加
    world_geom = trimesh.Trimesh(vertices=np.array([]), process=False)
    world_geom.metadata['name'] = 'world'
    world_geom.metadata['uuid'] = str(uuid.uuid4())
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

    # UUIDとノード名の双方向マッピング
    # trimeshがgeometryにnodeとは異なる名称を付与する場合があるのでトラックする目的
    scene.metadata['uuid_to_name'] = {}
    scene.metadata['uuid_to_name'][world_geom.metadata['uuid']] = world_geom.metadata['name']

    # シーンのメタデータに保存
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


def add_mesh(
        scene,
        name,
        mesh,
        position=None,
        parent_node=None):
    """
    trimesh メッシュをシーンに追加する

    Args:
        scene (trimesh.Scene): メッシュを追加するシーン
        name (str): メッシュの名前
        mesh (trimesh.Trimesh): メッシュ
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

    # metadata
    mesh.metadata['name'] = name

    # UUIDを付与してmeshに追加
    if 'uuid' not in mesh.metadata:
        mesh_uuid = str(uuid.uuid4())
        mesh.metadata['uuid'] = mesh_uuid
    mesh_uuid = mesh.metadata['uuid']

    # シーンにmeshを追加
    scene.add_geometry(mesh, node_name=name)
    if 'uuid_to_name' not in scene.metadata:
        scene.metadata['uuid_to_name'] = {}
    scene.metadata['uuid_to_name'][mesh_uuid] = name

    # 変換行列を準備
    # TODO: 引数にする
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


def create_mesh_triangle(
        name,
        vertices,
        faces,
        uvs,
        texture_path=None,
        material=None):
    """
    三角形のメッシュを作成する

    Args:
        name (str): メッシュの名前
        vertices (np.array): 頂点配列
        faces (np.array): 面配列
        uvs (np.array): UV座標配列
        texture_path (str): テクスチャ画像のパス
        material (Trimesh.Material): マテリアル

    Returns:
        trimesh.Trimesh: 作成したメッシュ
    """
    if texture_path is not None:
        # ビットマップをテクスチャ用に読み込む
        texture_img = Image.open(texture_path)
        # textureをもつメッシュを生成
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_img)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    else:
        if material is None:
            # TODO: デフォルトマテリアルを作成する
            pass


        # TODO: PBR Materialのテクスチャ設定が出来るようにする
        # 例:
        #   material = trimesh.visual.material.PBRMaterial(
        #       name="material_with_texture",
        #       # 基本的なPBRマテリアル属性
        #       metallicFactor=0.5,
        #       roughnessFactor=0.3,
        #       baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        #       emissiveFactor=[0.0, 0.0, 0.0],
        #       normalScale=1.0,
        #       occlusionStrength=1.0,
        # テクスチャマップの設定
        #       baseColorTexture=trimesh.visual.texture.Texture(
        #           image=texture_image,
        #           name="basecolor_texture"
        #       ),
        # 他のテクスチャマップも必要に応じて設定
        #       metallicRoughnessTexture=trimesh.visual.texture.Texture(...),
        #       normalTexture=trimesh.visual.texture.Texture(...),
        #       occlusionTexture=trimesh.visual.texture.Texture(...),
        #       emissiveTexture=trimesh.visual.texture.Texture(...)


        # PBRMaterialを使用する場合
        if isinstance(material, trimesh.visual.material.PBRMaterial):
            # face_colorsを設定（material.baseColorFactorを使用）
            color = material.baseColorFactor[:3]  # RGB部分だけ取得
            face_colors = np.array([color for _ in range(len(faces))], dtype=np.float32)

            # メッシュを生成してマテリアルを設定
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=face_colors,
                process=False
            )
            mesh.visual.material = material

        # SimpleMaterialを使用する場合
        elif isinstance(material, trimesh.visual.material.SimpleMaterial):
            # face_colorsを設定（material.diffuseを使用）
            color = material.diffuse
            face_colors = np.array([color for _ in range(len(faces))], dtype=np.float32)

            # メッシュを生成してマテリアルを設定
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=face_colors,
                process=False
            )
            mesh.visual.material = material

        # その他のマテリアルタイプ
        else:
            # デフォルトの色（灰色）
            face_colors = np.array([[0.5, 0.5, 0.5] for _ in range(len(faces))], dtype=np.float32)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                face_colors=face_colors,
                process=False
            )
            mesh.visual.material = material

    # metadata を作成
    mesh.metadata['name'] = name
    mesh_uuid = str(uuid.uuid4())
    mesh.metadata['uuid'] = mesh_uuid

    return mesh


def example_scene():
    """
    trimeshを使ってシーングラフを構築する（リファクタリング後のバージョン）

    Returns:
        trimesh.Scene: 構築されたシーン
    """
    # 空のシーンを作成
    scene = empty_scene()

    # triangle1を追加（親はworld）
    triangle1 = create_mesh_triangle(
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
    )
    add_mesh(scene, 'triangle1', triangle1, position=[0, 0, 0], parent_node='world')

    # triangle2を追加（親はtriangle1）
    triangle2 = create_mesh_triangle(
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
    )
    add_mesh(scene, 'triangle2', triangle2, position=[1.0, 1.0, 1.0], parent_node='triangle1')

    # triangle3を追加（PBRMaterialを使用、親はtriangle1）
    pbr_material = trimesh.visual.material.PBRMaterial(
        name='pbr_square',
        baseColorFactor=[1.0, 0.0, 0.0, 1.0],  # 赤色
        metallicFactor=0.8,
        roughnessFactor=0.2
    )

    triangle3 = create_mesh_triangle(
        name='triangle3',
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
        material=pbr_material,
    )
    add_mesh(scene, 'triangle3', triangle3, position=[-1.0, 0.0, 0.0], parent_node='triangle1')

    # triangle4を追加（SimpleMaterialを使用、親はtriangle1）
    simple_material = trimesh.visual.material.SimpleMaterial(
        diffuse=[0.0, 0.0, 1.0],  # 青色
        ambient=[0.1, 0.1, 0.1],
        specular=[1.0, 1.0, 1.0],
        glossiness=100.0
    )

    triangle4 = create_mesh_triangle(
        name='triangle4',
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
        material=simple_material,
    )
    add_mesh(scene, 'triangle4', triangle4, position=[0.0, -1.0, 0.0], parent_node='triangle1')

    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    # デバッグ出力
    print("\nManually defined hierarchy:")
    for node, parent in custom_hierarchy.items():
        print(f"Node: {node}, Parent: {parent}")

    print("\nManually defined transforms:")
    for node, transform in custom_transforms.items():
        if node == 'triangle2':
            print(f"Node: {node}, Transform:\n{transform}")

    return scene
