import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from build_scene import *


def make_empty_scene(output_path):
    # 空のシーンを作成
    scene = empty_scene()
    glb_path = convert_to_glb(scene, output_path)
    return glb_path


def make_example_scene(output_path):
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
        texture_path='../static/TestColorGrid.png',
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
        texture_path='../static/TestPicture.png',
    )
    add_mesh(scene, triangle2, position=[1.0, 1.0, 0.0], parent_geometry=triangle1)

    # triangle3を追加（PBRMaterialを使用、親はtriangle1）
    image_basecolor_1 = Image.open('../static/TestPicture.png')

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
    add_mesh(scene, triangle3, position=[-1.0, 1.0, 0.0], parent_geometry=triangle1)

    # triangle4を追加（PBRMaterialを使用、親はtriangle1）
    # TODO: metallicRoughness, nomal のテクスチャがどのようにGLTFに格納されるか調査
    #       現在、metallRoughness の B -> Metallic, G -> Roughtness となる. R -> occlusion ?
    #       glTF 2.0 spec: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
    image_1_grid_diffuse = Image.open('../static/TestColorGrid_diffuse.png')
    image_1_grid_rough   = Image.open('../static/TestColorGrid_rough.png')
    image_1_grid_normal  = Image.open('../static/TestColorGrid_normal.png')
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

    glb_path = convert_to_glb(scene, output_path)
    return glb_path


def main():

    # 空のシーン
    output_path = make_empty_scene("test_empty_scene.glb")

    # 複数の正方形、親子関係とマテリアルを設定
    output_path = make_example_scene("test_example_scene.glb")

    load_from_gltf_file("./test_empty_scene.gltf")

    scene = load_from_gltf_file("./test_example_scene.glb")
    subtree = get_node_subtree(scene, 'scene_root')

    def print_subtree(current, depth=0):
        node_name = current['node_name']
        geometry_name = current['geometry_name']
        children = current['children']
        indent = ' ' * depth
        print(f"{indent} node: {node_name}: geometry: {geometry_name}")
        for child in children:
            print_subtree(child, depth + 2)

    print_subtree(subtree)

    scene = example_scene(asset_root_path='../static/')
    asset_file = load_from_gltf_file("./TestAsset1.glb")
    asset_tree = get_node_subtree(asset_file, "Empty")
    print_subtree(asset_tree)

    add_subtree(scene, "triangle1", asset_tree)
    subtree = get_node_subtree(scene, 'scene_root')
    print_subtree(subtree)
    glb_path = convert_to_glb(scene, './merged_scene.glb')
    print(f" output: {glb_path}")

if __name__ == "__main__":
    main()
