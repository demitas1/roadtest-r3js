import trimesh
import numpy as np
from PIL import Image
import uuid

from .utils import compose_transform_matrix


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
    world_uuid = str(uuid.uuid4())
    world_geom.metadata['uuid'] = world_uuid
    scene.add_geometry(world_geom, node_name='world')

    # 親子関係 子uuid -> 親uuid
    custom_hierarchy = {
        world_uuid: None  # worldは親なし
    }

    # uuid -> 変換行列
    identity = np.eye(4)
    custom_transforms = {
        world_uuid: identity
    }

    # UUID->ノードのマッピング
    # trimeshがgeometryにnodeとは異なる名称を付与する場合があるのでトラックする目的
    scene.metadata['uuid_to_node'] = {}
    scene.metadata['uuid_to_node'][world_uuid] = world_geom

    # シーンのメタデータに保存
    scene.metadata['root_node'] = world_geom
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


def scene_root(scene):
    """
    シーンのルートノードを返す
    通常は world
    """
    return scene.metadata.get('root_node', None)


def add_mesh(
        scene,
        mesh,
        name=None,
        position=None,
        rotation=None,
        scale=None,
        parent_node=None):
    """
    trimesh メッシュをシーンに追加する

    Args:
        scene (trimesh.Scene): メッシュを追加するシーン
        mesh (trimesh.Trimesh): メッシュ
        name (str, optional): メッシュの名前(元の名前を上書きする場合に使用)
        position (list, optional): [x, y, z]の位置。Noneの場合は[0, 0, 0]
        parent_node (trimesh.Trimesh, optional): 親ノード。Noneの場合は'world'

    Returns:
        trimesh.Scene: 更新されたシーン
    """
    # デフォルト値の設定
    if position is None:
        position = [0, 0, 0]

    if parent_node is None:
        parent_node = scene_root(scene)
    parent_node_uuid = parent_node.metadata['uuid']

    # metadata
    if name is not None:
        # 元の名前を上書き
        node_name = name
        mesh.metadata['name'] = name
    else:
        # metadata内の名前を使用する
        node_name = mesh.metadata.get('name', None)

    # UUIDを付与してmeshに追加
    if 'uuid' not in mesh.metadata:
        mesh_uuid = str(uuid.uuid4())
        mesh.metadata['uuid'] = mesh_uuid
    mesh_uuid = mesh.metadata['uuid']

    # シーンにmeshを追加
    # TODO: metadataに辞書があることを保証できるようにする
    scene.add_geometry(mesh, node_name=node_name)
    scene.metadata['uuid_to_node'][mesh_uuid] = mesh

    # 変換行列を準備
    transform = compose_transform_matrix(
        translation=position,
        rotation=rotation,
        scale=scale)

    # 既存のメタデータを取得
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    # 親子関係と変換を更新
    custom_hierarchy[mesh_uuid] = parent_node_uuid
    custom_transforms[mesh_uuid] = transform

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

        # PBRMaterialを使用する場合
        elif isinstance(material, trimesh.visual.material.PBRMaterial):
            # メッシュを生成してマテリアルを設定
            visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=visual,
                process=False
            )

        # SimpleMaterialを使用する場合 (未対応)
        elif isinstance(material, trimesh.visual.material.SimpleMaterial):
            # メッシュを生成してマテリアルを設定
            visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=visual,
                process=False
            )

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

    # metadata に名前とuuidを作成
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
    add_mesh(scene, triangle1, name='triangle1', position=[0, 0, 0], parent_node=None)

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
    add_mesh(scene, triangle2, 'triangle2', position=[1.0, 1.0, 0.0], parent_node=triangle1)

    # triangle3を追加（PBRMaterialを使用、親はtriangle1）
    image_basecolor_1 = Image.open('./static/TestPicture.png')

    pbr_material_3 = trimesh.visual.material.PBRMaterial(
        name='pbr_square',
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.9,
        roughnessFactor=0.3,
        baseColorTexture=image_basecolor_1,
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
        material=pbr_material_3,
    )
    add_mesh(scene, triangle3, 'triangle3', position=[-1.0, 1.0, 0.0], parent_node=triangle1)

    # triangle4を追加（PBRMaterialを使用、親はtriangle1）
    # TODO: metallicRoughness, nomal のテクスチャがどのようにGLTFに格納されるか調査
    #       現在、metallRoughness の B -> Metallic, G -> Roughtness となる. R -> occlusion ?
    #       glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
    image_1_grid_diffuse = Image.open('./static/TestColorGrid_diffuse.png')
    image_1_grid_rough   = Image.open('./static/TestColorGrid_rough.png')
    image_1_grid_normal  = Image.open('./static/TestColorGrid_normal.png')
    pbr_material_4 = trimesh.visual.material.PBRMaterial(
        name='pbr_square',
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=1.0,
        roughnessFactor=1.0,
        baseColorTexture=image_1_grid_diffuse,
        metallicRoughnessTexture=image_1_grid_rough,
        normalTexture=image_1_grid_normal,
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
        material=pbr_material_4,
    )
    add_mesh(scene, triangle4, 'triangle4', position=[0.0, -1.0, 0.0], parent_node=triangle1)

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
