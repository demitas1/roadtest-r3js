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

    # root(empty) ジオメトリをシーンに追加
    # NOTE: trimeshのルートノードは 'world' というノード名のため衝突を避ける必要がある
    empty_geom = trimesh.Trimesh(vertices=np.array([]), process=False)
    empty_geom.metadata['name'] = 'scene_root'
    scene_root_uuid = str(uuid.uuid4())
    empty_geom.metadata['uuid'] = scene_root_uuid
    scene.add_geometry(empty_geom, node_name='world_root')

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


def get_geometry_from_node(scene, node_name):
    # ノードがジオメトリを持つかチェック
    if node_name in scene.graph.nodes_geometry:
        # scene.graph.get() でノードの変換行列とジオメトリ名を取得
        transform, geometry_name = scene.graph.get(node_name)
        return transform, geometry_name
    return None, None


def add_empty_geometry_to_nodes(scene, debug=False):
    pass
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

            geometry_name = f"empty:{node_name}:{geom_uuid}"  # geometry_name を一意にする
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

# TODO: add_mesh() のような add_group() 関数を作成する. 子ノード含めて追加
# TODO: scene, group, node (geometry, structure_node) の関係を整理

# TODO: scene_root() によって world, もしくは scene_root を返すように設計する
# NOTE: import の動作:
#       対象となる parent ノードを指定、その child に新規の empty ジオメトリを追加して scene_root 以下を移植する
#       (world)-(scene0_root)-(parent) <- (world)-(scene_root)-children (or (world)-children)
#       =>
#       (world)-(scene0_root)-(parent)-[empty]-children
#
#       インポートした後、empty の transform を操作することで children 全体を操作できるようにしたい


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
        parent_node (trimesh.Trimesh, optional): 親ノード。Noneの場合は scene_root

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
