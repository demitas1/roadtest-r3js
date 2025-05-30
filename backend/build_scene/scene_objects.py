import trimesh
import numpy as np
from PIL import Image
import uuid
import re

from .utils import compose_transform_matrix


def generate_unique_name(base_name, existing_names):
    """
    基本名から一意な名前を生成する（Blender方式）

    Args:
        base_name: 基本となる名前
        existing_names: 既存の名前のセット

    Returns:
        str: 一意な名前
    """
    if base_name not in existing_names:
        return base_name

    # 既存の枝番号を解析して最大値を見つける
    pattern = re.escape(base_name) + r'\.(\d{3})$'
    max_number = 0

    for existing_name in existing_names:
        match = re.match(pattern, existing_name)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)

    # 次の番号を生成（001, 002, ... 形式）
    next_number = max_number + 1
    return f"{base_name}.{next_number:03d}"


def get_geometry_from_node(scene, node_name):
    # ノードがあるかチェック
    if node_name in scene.graph.nodes:
        # scene.graph.get() でノードの変換行列とジオメトリ名を取得
        transform, geometry_name = scene.graph.get(node_name)
        return transform, geometry_name
    return None, None


def empty_scene():
    """
    Empty scene

    Returns:
        trimesh.Scene: empty trimesh Scene object
    """
    scene = trimesh.Scene()

    # root(empty) ジオメトリをシーンに追加
    # NOTE: trimeshのルートノードは 'world' というノード名のため衝突を避ける必要がある
    empty_geom = trimesh.Trimesh(vertices=np.array([]), process=False)
    empty_geom.metadata['name'] = 'scene_root'
    scene_root_uuid = str(uuid.uuid4())
    empty_geom.metadata['uuid'] = scene_root_uuid
    scene.add_geometry(empty_geom, node_name='scene_root')

    # 親子関係 子uuid -> 親uuid
    custom_hierarchy = {
        scene_root_uuid: None  # 親なし
    }

    # uuid -> 変換行列
    identity = np.eye(4)
    custom_transforms = {
        scene_root_uuid: identity
    }

    # UUID->ノードのマッピング
    # trimeshがgeometryにnodeとは異なる名称を付与する場合があるのでトラックする目的
    scene.metadata['uuid_to_node'] = {}
    scene.metadata['uuid_to_node'][scene_root_uuid] = empty_geom

    # シーンのメタデータに保存
    scene.metadata['scene_root_node'] = empty_geom
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    return scene


def scene_root(scene):
    """
    シーンのルートノードを返す
    通常は scene_root_node
    """
    # ルートノードがすでに設定されている場合はそれを返す
    root_node = scene.metadata.get('scene_root_node', None)
    if root_node:
        return root_node

    # uuidの親子辞書を使って, trimesh のルートノードを探す
    uuid_to_node = scene.metadata['uuid_to_node']
    custom_hierarchy = scene.metadata['custom_hierarchy']

    # 親がないノード、または自分自身が親となっているノードを見つける
    uuid_root_nodes = []
    for geom_key, geom in scene.geometry.items():
        geom_uuid = geom.metadata['uuid']
        parent_uuid = custom_hierarchy.get(geom_uuid, None)
        if parent_uuid is None or geom_uuid == parent_uuid:
            if geom_uuid not in uuid_root_nodes:
                uuid_root_nodes.append(geom_uuid)
    print(f" root geometry uuid: {uuid_root_nodes}")

    uuid_root_node = None
    if len(uuid_root_nodes) == 0:
        return None  # エラー (通常発生しないはず)
    elif len(uuid_root_nodes) == 1:
        uuid_root_node = uuid_root_nodes[0]
    else:
        # 複数あった場合は、'world'をルートとする
        for geom_uuid in uuid_root_nodes:
            geom = uuid_to_node[geom_uuid]
            print(f" uuid: {geom_uuid}: name: {geom.metadata['name']}")

    # 新たに見つけたルートノードを設定する
    root_geometry = uuid_to_node[uuid_root_node]
    root_geometry_name = root_geometry.metadata['name']
    root_node = scene.graph.geometry_nodes[root_geometry_name]
    print(f" uuid:{uuid_root_node}, geoemtry:{root_geometry_name}, node:{root_node}")
    scene.metadata['scene_root_node'] = root_node

    return root_node


def add_empty_geometry_to_nodes(scene, debug=False):
    """
    ジオメトリを持たないノードに空のジオメトリを付加する関数

    Args:
        scene: trimesh.Scene オブジェクト
        debug: デバッグ情報を出力するかどうか
    """
    # scene 既存のメタデータを取得
    uuid_to_node = scene.metadata.get('uuid_to_node', {})

    for node_name in scene.graph.nodes:
        transform, geometry_name = get_geometry_from_node(scene, node_name)
        if geometry_name is None:
            # 空のジオメトリを作成
            empty_geom = trimesh.Trimesh(vertices=np.array([]), process=False)

            # uuid を付加
            geom_uuid = str(uuid.uuid4())
            empty_geom.metadata['uuid'] = geom_uuid

            # node_name を元に, geometry_name を付与
            geometry_name = generate_unique_name(node_name, scene.geometry.keys())
            empty_geom.metadata['name'] = geometry_name

            # 既存のノードの変換行列を取得
            try:
                transform = scene.graph.get(node_name)[0]
            except:
                transform = np.eye(4)  # デフォルトの単位行列

            # 空のジオメトリをシーンに追加（既存のノード名を再利用）
            # 注意: ノードの辞書が更新されないように既存のノードを同名で上書きする
            scene.add_geometry(empty_geom, node_name=node_name, transform=transform)
            if debug:
                print(f"ノード '{node_name}' に空のジオメトリ {geometry_name} を追加しました")
            geom = empty_geom
        else:
            # ジオメトリに uuid を付加
            geom = scene.geometry.get(geometry_name)
            geom_uuid = str(uuid.uuid4())
            geom.metadata['uuid'] = geom_uuid

        # シーン内の uuid:geometry 辞書に追加
        uuid_to_node[geom_uuid] = geom

    # uuid:geometry 辞書を更新
    scene.metadata['uuid_to_node'] = uuid_to_node
    return scene


def update_scene_metadata(scene, debug=False):
    # scene 既存のメタデータを取得
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    # 親子関係の探索
    edges = scene.graph.to_edgelist()
    for edge in edges:
        parent_node = edge[0]
        child_node = edge[1]
        transform_parent, parent_geom = get_geometry_from_node(scene, parent_node)
        transform_child, child_geom = get_geometry_from_node(scene, child_node)
        if parent_geom is not None and child_geom is not None:
            parent_geometry = scene.geometry.get(parent_geom)
            child_geometry = scene.geometry.get(child_geom)

            parent_uuid = parent_geometry.metadata.get('uuid', None)
            child_uuid = child_geometry.metadata.get('uuid', None)
            if parent_uuid is not None and child_uuid is not None:
                custom_transforms[parent_uuid] = transform_parent
                custom_transforms[child_uuid] = transform_child
                custom_hierarchy[child_uuid] = parent_uuid

    # 親子関係と変換を更新
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms
    return scene


def load_from_gltf_file(gltf_file_path, world_node_name='world', debug=True):
    """
    GLTF/GLB ファイルから trimesh シーンを作成する
    各ノードに metadata, uuid を付与する
    """
    # glTF/glb ファイルを読み込む
    if debug:
        print(f"gltf file = {gltf_file_path}")
    scene = trimesh.load(gltf_file_path)

    if debug:
        print("=== empty 付加処理前の状態 ===")
        print(f"scene.graph.nodes: {list(scene.graph.nodes)}")
        print(f"scene.graph.nodes_geometry: {list(scene.graph.nodes_geometry)}")
        print(f"scene.graph.goemtry_nodes: {list(scene.graph.geometry_nodes)}")

    # ジオメトリを持たないノードに空のジオメトリを追加
    # NOTE: glTFからロードした場合、world, scene_root には geometry が存在しない
    scene = add_empty_geometry_to_nodes(scene, debug=debug)
    # 全ノードに uuid を付与し、シーン内の親子関係を更新
    scene = update_scene_metadata(scene, debug=debug)

    if debug:
        print("=== empty 付加処理後の状態 ===")
        print(f"scene.graph.nodes: {list(scene.graph.nodes)}")
        print(f"scene.graph.nodes_geometry: {list(scene.graph.nodes_geometry)}")
        print(f"scene.graph.goemtry_nodes: {list(scene.graph.geometry_nodes)}")

    # ルートノードを見つけて設定する
    root_node = scene_root(scene)

    return scene


def get_node_subtree(scene, node_name):
    """
    指定されたノードとその子孫ノードの情報を取得

    Args:
        scene: trimesh Scene オブジェクト
        node_name: ルートとなるノード名

    Returns:
        dict: ノード情報（ノード、ジオメトリ、変換行列、子ノード）
    """
    if node_name not in scene.graph.nodes:
        return None

    # エッジリストを取得
    edges = scene.graph.to_edgelist()

    def get_children(node):
        """指定されたノードの子ノードを取得"""
        children = []
        for edge in edges:
            if len(edge) >= 2 and edge[0] == node and edge[0] != edge[1]:
                children.append(edge[1])
        return children

    def collect_subtree(current_node):
        """
        再帰的にサブツリーを収集
        """
        node_info = {
            'node_name': current_node,
            'transform': None,
            'geometry_name': None,
            'geometry': None,
            'children': []
        }

        # ジオメトリ情報を取得
        transform, geometry_name = get_geometry_from_node(scene, current_node)
        node_info['transform'] = transform
        if geometry_name is not None:
            if geometry_name in scene.geometry:
                node_info['geometry_name'] = geometry_name
                node_info['geometry'] = scene.geometry[geometry_name]

        # 子ノードを再帰的に処理
        children = get_children(current_node)
        for child in children:
            child_info = collect_subtree(child)
            if child_info:
                node_info['children'].append(child_info)

        return node_info

    return collect_subtree(node_name)


# TODO: scene, group, node (geometry, structure_node) の関係を整理
# NOTE: import の動作:
#       対象となる parent ノードを指定、その child に新規の empty ジオメトリを追加して scene_root 以下を移植する
#       (world)-(scene0_root)-(parent) <- (world)-(scene_root)-children (or (world)-children)
#       =>
#       (world)-(scene0_root)-(parent)-[empty]-children
#
#       インポートした後、empty の transform を操作することで children 全体を操作できるようにしたい
def add_subtree(target_scene, target_parent_geometry_name, subtree):
    """
    サブツリーをターゲットシーンに追加

    Args:
        target_scene: 追加先のシーン
        target_parent_node: 親ノード名
        subtree: 追加するサブツリー情報

    Returns:
        str: 追加されたルートノードの名前
    """
    # ノード名を生成（重複回避）
    original_node_name = subtree['node_name']
    original_geometry_name = subtree['geometry_name']

    # 変換行列を用意
    transform = subtree['transform']
    if transform is None:
        transform = np.eye(4, dtype=np.float64)

    # ジオメトリを追加
    geometry_to_add = subtree['geometry']
    if geometry_to_add is None:
        # ジオメトリなしのノードの場合、empty メッシュを作成
        geometry_to_add = trimesh.Trimesh(vertices=np.array([]), process=False)

    # target_parent_geometry の存在を確認
    target_parent_geometry = target_scene.geometry.get(target_parent_geometry_name, None)
    if target_parent_geometry is None:
        print(f"Error: failed to find parent geometry:{target_parent_geometry_name}.")
        return None

    # シーンに追加
    new_node_name = add_mesh(
            target_scene,
            geometry_to_add,
            node_name=original_node_name,
            geometry_name=original_geometry_name,
            transform=transform,
            parent_geometry=target_parent_geometry,
            clone=True)

    # ノードとジオメトリが追加されたことを確認する
    if new_node_name is None:
        print(f"Error: failed to add new node {original_node_name}:{original_geometry_name}.")
        return None

    _, new_geometry_name = get_geometry_from_node(target_scene, new_node_name)
    if new_geometry_name is None:
        print(f"Error: failed to add new node {original_node_name}:{original_geometry_name}.")
        return None

    # 子ノードを再帰的に追加
    for child_info in subtree['children']:
        add_subtree(target_scene, new_geometry_name, child_info)

    return new_node_name


# TODO: 返り値が適切か再考. sceneは不要. 実際に付与されたメッシュに関する情報の方が重要.
def add_mesh(
        scene,
        mesh_to_add,
        node_name=None,
        geometry_name=None,
        transform=None,
        position=None,
        rotation=None,
        scale=None,
        parent_geometry=None,
        clone=True):
    """
    trimesh メッシュをシーンに追加する

    Args:
        scene (trimesh.Scene): メッシュを追加するシーン
        mesh (trimesh.Trimesh): メッシュ
        node_name (str, optional): シーングラフ内の node の名前 (デフォルト: geometry の名前を使用)
        geometry_name (str, optional): mesh geometry の名前 (元の名前を上書きする場合に使用)
        transform (np.array): 4x4 numpy行列. transformを指定した場合は position, rotation, scale は無視される
        position (list, optional): [x, y, z]の位置。Noneの場合は[0, 0, 0]
        rotation (list, optional): 回転クォータニオン [x, y, z, w]
        scale (list, optional): スケール [x, y, z]
        parent_geometry (trimesh.Trimesh, optional): 親ノードのジオメトリ。Noneの場合は scene_root
        clone (boolean): mesh を clone してからシーンに追加する.

    Returns:
        str: 追加されたノードの名称
    """
    # 追加先シーンの uuid:geometry 辞書
    uuid_to_node = scene.metadata.get('uuid_to_node', {})

    # 親ノードが無指定の場合シーンのルートノードを親とする
    if parent_geometry is None:
        parent_geometry = scene_root(scene)
    parent_node_uuid = parent_geometry.metadata['uuid']

    # 親ノードが追加先シーンにない場合、警告を出しルートノードを親とする
    if parent_node_uuid not in uuid_to_node:
        print(f"Warning: parent node uuid {parent_node_uuid} not in the scene.")
        parent_geometry = scene_root(scene)
        parent_node_uuid = parent_geometry.metadata['uuid']

    # 親ノードのジオメトリ名、ノード名を取得
    parent_geometry_name = parent_geometry.metadata['name']
    # TODO: trimeshでは一つのジオメトリが複数のノードで共有される場合があるが、ここでは1:1対応であることを強制する
    parent_nodes = scene.graph.geometry_nodes[parent_geometry_name]
    parent_node_name = parent_nodes[0]
    print(f" add_mesh: parent node:geometry = {parent_node_name}:{parent_geometry_name}")

    # Trimesh.copy() をつかってクローンを add する
    if clone:
        mesh = mesh_to_add.copy()
    else:
        mesh = mesh_to_add

    # UUID を追加する mesh に付与
    if 'uuid' not in mesh.metadata:
        mesh_uuid = str(uuid.uuid4())
        mesh.metadata['uuid'] = mesh_uuid
    mesh_uuid = mesh.metadata['uuid']

    # geometry name
    if geometry_name is None:
        # 無指定の場合 metadata の名前を使用する
        geometry_name = mesh.metadata.get('name', None)
    # 追加先のシーン内に重複がある場合, 別名を与える
    geometry_unique_name = generate_unique_name(geometry_name, scene.geometry.keys())
    mesh.metadata['name'] = geometry_unique_name

    # node name
    if node_name is None:
        node_name = geometry_unique_name
    # 追加先のシーン内に重複がある場合, 別名を与える
    node_unique_name = generate_unique_name(node_name, scene.graph.nodes)

    # 変換行列を準備
    # TODO: transform引数の型チェック. 不正な場合は単位行列をセットする.
    if transform is None:
        transform = compose_transform_matrix(
            translation=position,
            rotation=rotation,
            scale=scale)

    # シーンにmeshを追加
    # TODO: 戻り値をチェック. None ならエラー
    new_node = scene.add_geometry(
        mesh,
        node_name=node_unique_name,
        geom_name=geometry_unique_name,
        transform=transform,
        parent_node_name=parent_node_name)

    # uuid:gemetry 辞書を更新
    uuid_to_node[mesh_uuid] = mesh
    scene.metadata['uuid_to_node'] = uuid_to_node

    # 既存のメタデータを取得
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    # 親子関係と変換を更新
    custom_hierarchy[mesh_uuid] = parent_node_uuid
    custom_transforms[mesh_uuid] = transform

    # シーンのメタデータに保存し直す
    scene.metadata['custom_hierarchy'] = custom_hierarchy
    scene.metadata['custom_transforms'] = custom_transforms

    # 実際に付与された名前の情報を返す
    return new_node


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


# TODO: asset_root_path を別関数で処理

def example_scene(asset_root_path='./static/'):
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
        texture_path=asset_root_path + 'TestColorGrid.png',
    )
    add_mesh(scene, triangle1, position=[0, 0, 0], parent_geometry=None)

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
        texture_path=asset_root_path + 'TestPicture.png',
    )
    add_mesh(scene, triangle2, position=[1.0, 1.0, 0.0], parent_geometry=triangle1)

    # triangle3を追加（PBRMaterialを使用、親はtriangle1）
    image_basecolor_1 = Image.open(asset_root_path + 'TestPicture.png')

    pbr_material_3 = trimesh.visual.material.PBRMaterial(
        name='pbr_square',
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.9,
        roughnessFactor=0.3,
        baseColorTexture=image_basecolor_1,
    )

    triangle3 = create_mesh_triangle(
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
        material=pbr_material_3,
    )
    add_mesh(scene, triangle3, position=[-1.0, 1.0, 0.0], parent_geometry=triangle1)

    # triangle4を追加（PBRMaterialを使用、親はtriangle1）
    # TODO: metallicRoughness, nomal のテクスチャがどのようにGLTFに格納されるか調査
    #       現在、metallRoughness の B -> Metallic, G -> Roughtness となる. R -> occlusion ?
    #       glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
    image_1_grid_diffuse = Image.open(asset_root_path + 'TestColorGrid_diffuse.png')
    image_1_grid_rough   = Image.open(asset_root_path + 'TestColorGrid_rough.png')
    image_1_grid_normal  = Image.open(asset_root_path + 'TestColorGrid_normal.png')
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
        material=pbr_material_4,
    )
    add_mesh(scene, triangle4, position=[0.0, -1.0, 0.0], parent_geometry=triangle1)

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
