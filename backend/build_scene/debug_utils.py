import trimesh
from trimesh.scene import Scene
import numpy as np
import copy


# TODO: デバッグ用途のもののみにする
# TODO: その他の有用な関数は utils.py へ


def find_node_by_name(scene, target_name):
    """
    シーン内から指定された名前のノードを検索

    Args:
        scene: trimesh Scene オブジェクト
        target_name: 検索するノード名

    Returns:
        str or None: 見つかったノード名、見つからない場合はNone
    """
    for node in scene.graph.nodes:
        if node == target_name:
            return node
    return None

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
            if len(edge) >= 2 and edge[0] == node:
                children.append(edge[1])
        return children

    def collect_subtree(current_node):
        """
        再帰的にサブツリーを収集
        """
        node_info = {
            'node_name': current_node,
            'transform': scene.graph.get(current_node),
            'geometry_name': None,
            'geometry': None,
            'children': []
        }

        # ジオメトリ情報を取得
        if current_node in scene.graph.nodes_geometry:
            _, geometry_name = scene.graph[current_node]
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

def generate_unique_name(scene, base_name):
    """
    シーン内でユニークな名前を生成

    Args:
        scene: trimesh Scene オブジェクト
        base_name: ベースとなる名前

    Returns:
        str: ユニークな名前
    """
    if base_name not in scene.graph.nodes and base_name not in scene.geometry:
        return base_name

    counter = 1
    while True:
        new_name = f"{base_name}_{counter}"
        if new_name not in scene.graph.nodes and new_name not in scene.geometry:
            return new_name
        counter += 1

def add_subtree_to_scene(target_scene, parent_node, subtree_info, name_prefix=""):
    """
    サブツリーをターゲットシーンに追加

    Args:
        target_scene: 追加先のシーン
        parent_node: 親ノード名
        subtree_info: 追加するサブツリー情報
        name_prefix: 名前の接頭辞

    Returns:
        str: 追加されたルートノードの名前
    """
    # ノード名を生成（重複回避）
    original_name = subtree_info['node_name']
    new_node_name = generate_unique_name(target_scene, f"{name_prefix}{original_name}")

    # 変換行列を適切な形式に変換
    transform = subtree_info['transform']
    if transform is not None:
        # numpy配列として確実に変換し、4x4行列にする
        if hasattr(transform, 'shape'):
            if transform.shape == (4, 4):
                transform = np.array(transform, dtype=np.float64)
            else:
                # 4x4でない場合は単位行列を使用
                transform = np.eye(4, dtype=np.float64)
        else:
            # transformがない場合は単位行列
            transform = np.eye(4, dtype=np.float64)
    else:
        transform = np.eye(4, dtype=np.float64)

    # ジオメトリがある場合は追加
    if subtree_info['geometry'] is not None:
        # ジオメトリをコピー
        geometry_copy = copy.deepcopy(subtree_info['geometry'])

        # add_geometryメソッドを使用してノードとジオメトリを同時に追加
        target_scene.add_geometry(
            geometry=geometry_copy,
            node_name=new_node_name,
            parent_node_name=parent_node,
            transform=transform
        )
    else:
        # ジオメトリなしのノードの場合、小さな空のメッシュを作成
        empty_mesh = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0]], dtype=np.float64), 
            faces=np.array([], dtype=np.int32).reshape(0, 3)
        )
        target_scene.add_geometry(
            geometry=empty_mesh,
            node_name=new_node_name,
            parent_node_name=parent_node,
            transform=transform
        )

    # 子ノードを再帰的に追加
    for child_info in subtree_info['children']:
        add_subtree_to_scene(target_scene, new_node_name, child_info, name_prefix)

    return new_node_name

def transfer_node_between_scenes(scene0, node0_name, scene1, node1_name, name_prefix="imported_"):
    """
    scene1のnode1をscene0のnode0の子として移植

    Args:
        scene0: 移植先のシーン
        node0_name: 移植先の親ノード名
        scene1: 移植元のシーン
        node1_name: 移植元のノード名
        name_prefix: 移植時の名前接頭辞

    Returns:
        tuple: (成功フラグ, 追加されたノード名 or エラーメッセージ)
    """
    # scene0内のnode0を確認
    if node0_name not in scene0.graph.nodes:
        return False, f"Node '{node0_name}' not found in target scene"

    # scene1内のnode1を検索
    found_node1 = find_node_by_name(scene1, node1_name)
    if not found_node1:
        return False, f"Node '{node1_name}' not found in source scene"

    # node1のサブツリーを取得
    subtree_info = get_node_subtree(scene1, found_node1)
    if not subtree_info:
        return False, f"Failed to extract subtree for node '{node1_name}'"

    try:
        # サブツリーをscene0に追加
        new_root_node = add_subtree_to_scene(scene0, node0_name, subtree_info, name_prefix)
        return True, new_root_node
    except Exception as e:
        return False, f"Failed to transfer node: {str(e)}"


# TODO: uuidを使ったtrimeshシーンに対応して、もっとシンプルにする
def print_scene_structure_with_material(scene, title="Scene Structure"):
    """
    シーン構造をシンプルに表示（階層インデント付き、マテリアル・テクスチャ情報含む）
    """
    print(f"\n=== {title} ===")

    # エッジリストを取得
    edges = scene.graph.to_edgelist()

    # 全ノードから親を持たないルートノードを見つける
    all_nodes = set(scene.graph.nodes)
    child_nodes = set()

    # 親子関係を調べて子ノードを特定
    for edge in edges:
        if len(edge) >= 2:
            parent, child = edge[0], edge[1]
            child_nodes.add(child)

    # ルートノード = 全ノード - 子ノード
    root_nodes = all_nodes - child_nodes

    def get_children(node):
        """指定されたノードの子ノードを取得"""
        children = []
        for edge in edges:
            if len(edge) >= 2 and edge[0] == node:
                children.append(edge[1])
        return children

    def get_material_info(geometry):
        """ジオメトリからマテリアル情報を取得"""
        if not hasattr(geometry, 'visual') or geometry.visual is None:
            return None, []

        visual = geometry.visual
        if not hasattr(visual, 'material') or visual.material is None:
            return None, []

        material = visual.material
        material_type = type(material).__name__

        # テクスチャ情報を収集
        textures = []
        texture_properties = [
            'baseColorTexture',
            'metallicRoughnessTexture',
            'normalTexture',
            'occlusionTexture',
            'emissiveTexture'
        ]

        for tex_prop in texture_properties:
            if hasattr(material, tex_prop):
                texture = getattr(material, tex_prop)
                if texture is not None:
                    # プロパティ名から表示用の名前を作成
                    display_name = tex_prop.replace('Texture', '').replace('baseColor', 'diffuse')
                    textures.append(display_name)

        return material_type, textures

    def traverse_and_print(node, depth=0):
        """ノードを再帰的に巡回して表示"""
        indent = "  " * depth
        print(f"{indent}{node}", end="")

        # ノードがジオメトリを持つか確認
        if node in scene.graph.nodes_geometry:
            _, geometry_name = scene.graph[node]
            if geometry_name in scene.geometry:
                geometry = scene.geometry[geometry_name]

                # メッシュかどうか確認
                if hasattr(geometry, 'vertices') and hasattr(geometry, 'faces'):
                    print(f" [mesh]", end="")

                    # マテリアル情報を取得
                    material_type, textures = get_material_info(geometry)

                    if material_type:
                        print(f" [material: {material_type}]", end="")

                        if textures:
                            tex_str = ", ".join(textures)
                            print(f" [textures: {tex_str}]", end="")

        print()  # 改行

        # 子ノードを取得して再帰的に処理
        children = get_children(node)
        for child in children:
            traverse_and_print(child, depth + 1)

    # 各ルートノードから巡回開始
    for root in root_nodes:
        traverse_and_print(root)

# 使用例
def example_usage():
    """
    使用例のデモンストレーション
    """
    print("=== Node Transfer Example ===")

    # scene0を作成（移植先）
    scene0 = Scene()

    # 基本的なジオメトリを追加
    box = trimesh.creation.box(extents=[1, 1, 1])
    scene0.add_geometry(box, node_name="root_node")
    scene0.add_geometry(box, node_name="target_parent", parent_node_name="root_node")

    print("Initial scene0 structure:")
    print_scene_structure_with_material(scene0, "Scene0 (Before)")

    # scene1をGLTFファイルから読み込み（実際の使用時）
    # scene1 = trimesh.load("path/to/your/file.gltf")

    # デモ用にscene1を作成
    scene1 = Scene()
    sphere = trimesh.creation.icosphere(radius=0.5)
    cylinder = trimesh.creation.cylinder(radius=0.3, height=1.0)

    scene1.add_geometry(sphere, node_name="imported_root")
    scene1.add_geometry(cylinder, node_name="imported_child", parent_node_name="imported_root")

    print_scene_structure_with_material(scene1, "Scene1 (Source)")

    # ノード移植を実行
    success, result = transfer_node_between_scenes(
        scene0=scene0,
        node0_name="target_parent",
        scene1=scene1,
        node1_name="imported_root",
        name_prefix="from_gltf_"
    )

    if success:
        print(f"\n✅ Transfer successful! New node: {result}")
        print_scene_structure_with_material(scene0, "Scene0 (After)")
    else:
        print(f"\n❌ Transfer failed: {result}")

    return scene0, scene1


def transfer_from_gltf_file(scene0, node0_name, gltf_file_path, node1_name, name_prefix="gltf_"):
    """
    GLTFファイルから直接ノードを移植する便利関数

    Args:
        scene0: 移植先のシーン
        node0_name: 移植先の親ノード名
        gltf_file_path: GLTFファイルのパス
        node1_name: 移植するノード名
        name_prefix: 名前接頭辞

    Returns:
        tuple: (成功フラグ, 結果メッセージ)
    """
    try:
        # GLTFファイルを読み込み
        scene1 = trimesh.load(gltf_file_path)

        # シーンオブジェクトでない場合は変換
        if not isinstance(scene1, Scene):
            scene1 = Scene(scene1)

        # インポートするシーンの内容を表示
        print_scene_structure_with_material(scene1, title="scene to import")

        # ノード移植を実行
        return transfer_node_between_scenes(scene0, node0_name, scene1, node1_name, name_prefix)

    except Exception as e:
        return False, f"Failed to load GLTF file: {str(e)}"

# メイン実行部分
if __name__ == "__main__":
    # 使用例を実行
    if False:
        scene0, scene1 = example_usage()

    # 実際のGLTFファイルを使用する場合の例
    # scene0を準備
    scene0 = Scene()
    box = trimesh.creation.box(extents=[2, 2, 2])
    scene0.add_geometry(box, node_name="main_object")
    print_scene_structure_with_material(scene0, "Scene0 (Source)")

    # GLTFファイルからノードを移植
    success, result = transfer_from_gltf_file(
        scene0=scene0,
        node0_name="main_object",  # 'main_object' の child としてインポートする場合
        #node0_name="world",       # 'world' (ルートノード) の child としてインポートする場合
        gltf_file_path="./TestCube.glb",
        node1_name="Cube",
        name_prefix="imported_"
    )

    if success:
        print(f"Successfully imported node: {result}")
        print_scene_structure_with_material(scene0, "Scene0 (After)")
    else:
        print(f"Import failed: {result}")
