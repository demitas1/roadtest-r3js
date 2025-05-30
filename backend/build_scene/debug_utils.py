import trimesh
from trimesh.scene import Scene
import numpy as np


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