import bpy
import sys

# コマンドライン引数からモデルのパスを取得
model_path = sys.argv[sys.argv.index('--') + 1]

# インポート前にデフォルトのオブジェクトを削除
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# モデルをインポート
bpy.ops.import_scene.gltf(filepath=model_path)

# マテリアルの一覧を取得
print("\n=== マテリアル一覧 ===")
for idx, mat in enumerate(bpy.data.materials):
    print(f"Material[{idx}]: {mat.name}")

    # マテリアルのプロパティを表示
    if mat.use_nodes:
        print(f"  ノードベースのマテリアル")

        # プリンシプルBSDFノードを探す
        principled_bsdf = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled_bsdf = node
                # 必ず存在するパラメータ
                print(f"  Base Color: {node.inputs['Base Color'].default_value[:3]}")
                print(f"  Metallic: {node.inputs['Metallic'].default_value}")
                print(f"  Roughness: {node.inputs['Roughness'].default_value}")
                print(f"  IOR: {node.inputs['IOR'].default_value}")

                # ノーマルマップの入力を確認
                normal_input = node.inputs.get('Normal')
                if normal_input and normal_input.is_linked:
                    print(f"  Normal Map: Connected")
                else:
                    print(f"  Normal Map: Not Connected")

                # 他のパラメータを安全に取得
                # Blender 4.xでの入力名の変更に対応
                for input_name in ['Specular', 'Specular IOR Level', 'Specular Tint', 'Transmission', 'Transmission Roughness', 'Alpha']:
                    if input_name in node.inputs:
                        print(f"  {input_name}: {node.inputs[input_name].default_value}")

        # すべてのノードとリンクの接続を調査
        normal_map_node = None
        for node in mat.node_tree.nodes:
            if node.type == 'NORMAL_MAP':
                normal_map_node = node
                print(f"  Found Normal Map Node: {node.name}")
                if 'Strength' in node.inputs:
                    print(f"    Strength: {node.inputs['Strength'].default_value}")

        # テクスチャノードを探す
        print(f"  テクスチャ:")
        texture_found = False
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                texture_found = True
                # このテクスチャが接続されている入力を探す
                connected_to = []
                for link in mat.node_tree.links:
                    if link.from_node == node:
                        # テクスチャの接続先を探る
                        to_node = link.to_node
                        if to_node.type == 'BSDF_PRINCIPLED':
                            input_name = link.to_socket.name if hasattr(link.to_socket, 'name') else "Unknown"
                            connected_to.append(input_name)
                        elif to_node.type == 'NORMAL_MAP':
                            connected_to.append("Normal Map")
                        else:
                            input_name = link.to_socket.name if hasattr(link.to_socket, 'name') else "Unknown"
                            connected_to.append(f"{to_node.name}:{input_name}")

                connection_info = f" -> {', '.join(connected_to)}" if connected_to else ""
                print(f"    Image: {node.image.name}{connection_info}")
                print(f"      Size: {node.image.size[0]}x{node.image.size[1]}")
                if node.image.filepath:
                    print(f"      Filepath: {node.image.filepath}")
                else:
                    print(f"      Generated texture")

                # テクスチャのカラースペースを表示
                if hasattr(node.image, 'colorspace_settings'):
                    print(f"      Color Space: {node.image.colorspace_settings.name}")

        if not texture_found:
            print("    No textures found")
    else:
        print("  非ノードベースのマテリアル")
        print(f"  Diffuse Color: {mat.diffuse_color[:3]}")

# シーン内のすべてのオブジェクトの親子関係を解析する関数
def print_hierarchy(obj, level=0):
    indent = "  " * level
    print(f"{indent}Object: {obj.name} (Type: {obj.type})")

    # トランスフォーム情報
    print(f"{indent}  Location: {obj.location[:]}") 
    print(f"{indent}  Rotation: {[round(r, 2) for r in obj.rotation_euler[:]]}") 
    print(f"{indent}  Scale: {obj.scale[:]}") 

    # メッシュ情報
    if obj.type == 'MESH':
        print(f"{indent}  Vertices: {len(obj.data.vertices)}")
        print(f"{indent}  Faces: {len(obj.data.polygons)}")

        # UV情報
        if obj.data.uv_layers:
            print(f"{indent}  UV Layers:")
            for uv_layer in obj.data.uv_layers:
                print(f"{indent}    {uv_layer.name}")

        # マテリアル情報
        if obj.material_slots:
            print(f"{indent}  Materials:")
            for slot in obj.material_slots:
                if slot.material:
                    print(f"{indent}    {slot.material.name}")

    # カメラ情報
    elif obj.type == 'CAMERA':
        print(f"{indent}  Lens: {obj.data.lens}mm")
        print(f"{indent}  Sensor Width: {obj.data.sensor_width}mm")

    # ライト情報
    elif obj.type == 'LIGHT':
        print(f"{indent}  Type: {obj.data.type}")
        print(f"{indent}  Energy: {obj.data.energy}")
        print(f"{indent}  Color: {obj.data.color[:3]}")

    # 子オブジェクトを再帰的に処理
    for child in obj.children:
        print_hierarchy(child, level + 1)

# 親のないルートオブジェクトを取得してツリー表示
print("\n=== オブジェクト階層 ===")
for obj in bpy.context.scene.objects:
    if not obj.parent:
        print_hierarchy(obj)

# アーマチュアがある場合はボーン階層も表示
print("\n=== ボーン階層 ===")
armature_found = False
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE':
        armature_found = True
        print(f"Armature: {obj.name}")

        def print_bone_hierarchy(bone, level=0):
            indent = "  " * level
            print(f"{indent}Bone: {bone.name}")
            print(f"{indent}  Head: {bone.head_local[:]}")
            print(f"{indent}  Tail: {bone.tail_local[:]}")
            print(f"{indent}  Length: {bone.length}")
            for child in bone.children:
                print_bone_hierarchy(child, level + 1)

        for bone in obj.data.bones:
            if not bone.parent:
                print_bone_hierarchy(bone)

if not armature_found:
    print("  No armatures found")

# アニメーション情報の表示
print("\n=== アニメーション ===")
if bpy.data.actions:
    for action in bpy.data.actions:
        print(f"Action: {action.name}")
        print(f"  Frame Range: {int(action.frame_range[0])} to {int(action.frame_range[1])}")
        print(f"  Channels: {len(action.fcurves)}")
else:
    print("  No animations found")

# glTFエクストラデータがあれば表示
print("\n=== glTF エクストラデータ ===")
extra_data_found = False
for obj in bpy.context.scene.objects:
    if hasattr(obj, 'gltf_extras'):
        extra_data_found = True
        print(f"Object {obj.name} extras: {obj.gltf_extras}")

if not extra_data_found:
    print("  No glTF extra data found")
