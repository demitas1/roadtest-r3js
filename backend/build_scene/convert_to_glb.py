import trimesh
import numpy as np
from PIL import Image
import struct
import io
import os
import json
from pygltflib import GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes
from pygltflib import Buffer, BufferView, Accessor, Material, PbrMetallicRoughness
from pygltflib import Texture, Sampler, Image as GLTFImage, TextureInfo

from .constants import *


def convert_to_glb(scene, output_path="./static/output.glb", debug=True):
    """
    trimeshのシーンをGLBファイルに変換する

    Args:
        scene (trimesh.Scene): 変換するtrimeshシーン
        output_path (str): 出力するGLBファイルのパス
        debug (bool): デバッグ情報を出力するかどうか

    Returns:
        {
            'glb_path': 保存されたGLBファイルのパス
        }
    """
    # カスタム階層情報を取得（存在する場合）
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})

    if debug:
        print("\nConverting using custom hierarchy:")
        for node, parent in custom_hierarchy.items():
            print(f"Node: {node}, Parent: {parent}")

        print("\nScene graph nodes:")
        for node in scene.graph.nodes:
            print(f"- {node}")

        print("\nScene graph nodes_geometry:")
        for node in scene.graph.nodes_geometry:
            print(f"- {node}")

        print("\nScene geometry keys:")
        for key in scene.geometry.keys():
            print(f"- {key}")

    # シーンからメッシュ情報を取得
    meshes = {}
    mesh_names = []  # メッシュ名のリストを保持する配列
    texture_images = {}  # メッシュ名をキーとした画像データを保持する辞書

    # 構造ノード（メッシュがないノード）のセット
    structure_nodes = set()

    for node_name in scene.graph.nodes_geometry:
        if node_name in scene.geometry:
            mesh = scene.geometry[node_name]

            # 頂点座標、面情報、UV座標を取得
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces

            # 空のメッシュの場合は構造ノードとして扱う
            if len(vertices) == 0 or len(faces) == 0:
                if debug:
                    print(f"Info: Node '{node_name}' has no geometry, treating as structure node")
                structure_nodes.add(node_name)
                continue

            # メッシュ名を追加
            mesh_names.append(node_name)

            # UV座標の取得
            if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uvs = mesh.visual.uv.astype(np.float32)
            else:
                # UVがない場合は頂点数分のデフォルト値を作成
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)

            # テクスチャ画像の取得
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
                image = mesh.visual.material.image
                texture_images[node_name] = image
            elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'texture'):
                # textureプロパティからイメージを取得
                texture = mesh.visual.material.texture
                if hasattr(texture, 'image'):
                    texture_images[node_name] = texture.image

            meshes[node_name] = {
                'vertices': vertices,
                'faces': faces,
                'uvs': uvs,
                'mesh_obj': mesh
            }

    # シーングラフの他のノード（メッシュのないノード）も追加
    for node_name in scene.graph.nodes:
        if node_name not in scene.graph.nodes_geometry:
            structure_nodes.add(node_name)

    if debug:
        print("\nStructure nodes (non-mesh):")
        for node in structure_nodes:
            print(f"- {node}")

        print("\nMesh nodes:")
        for node in mesh_names:
            print(f"- {node}")

    # メッシュノードが存在するか確認
    if not mesh_names:
        if debug:
            print("No valid meshes found in the scene. Creating a nodes-only GLTF file.")

        # メッシュが存在しない場合、pygltflibを使わずに直接GLTFを作成
        # これにより、バッファなしの純粋なノード構造のGLTFを生成

        # GLTF構造を手動で構築
        gltf_json = {
            "asset": {
                "version": "2.0"
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Scene",
                    "nodes": []  # ルートノードのインデックスをここに追加
                }
            ],
            "nodes": []  # ノード情報をここに追加
        }

        # ノード情報を追加
        nodes_dict = {}  # ノード名とインデックスのマッピング
        node_index = 0

        for node_name in structure_nodes:
            # カスタム変換行列を取得
            if node_name in custom_transforms:
                transform = custom_transforms[node_name]
            else:
                transform = scene.graph[node_name][0] if node_name in scene.graph else np.eye(4)

            # 変換を分解: 平行移動、回転、スケールに
            translation = transform[:3, 3].tolist()

            # デフォルトの回転とスケール
            rotation = [0.0, 0.0, 0.0, 1.0]
            scale = [1.0, 1.0, 1.0]

            # ノード情報を追加
            node_info = {
                "name": node_name,
                "translation": translation,
                "rotation": rotation,
                "scale": scale
            }

            gltf_json["nodes"].append(node_info)
            nodes_dict[node_name] = node_index
            node_index += 1

        # 親子関係を設定
        for node_name, parent_name in custom_hierarchy.items():
            if parent_name is not None and parent_name in nodes_dict and node_name in nodes_dict:
                parent_index = nodes_dict[parent_name]
                child_index = nodes_dict[node_name]

                # 親ノードの子リストを更新
                if "children" not in gltf_json["nodes"][parent_index]:
                    gltf_json["nodes"][parent_index]["children"] = []

                # 既に子リストに追加されていない場合のみ追加
                if child_index not in gltf_json["nodes"][parent_index]["children"]:
                    gltf_json["nodes"][parent_index]["children"].append(child_index)
                    if debug:
                        print(f"Added node '{node_name}' (index {child_index}) as child of '{parent_name}' (index {parent_index})")

        # ルートノードをシーンに追加
        root_nodes = []
        for node_name in structure_nodes:
            if node_name in custom_hierarchy and custom_hierarchy[node_name] is None:
                if node_name in nodes_dict:
                    root_node_index = nodes_dict[node_name]
                    root_nodes.append(root_node_index)
                    if debug:
                        print(f"Added root node: '{node_name}' (index {root_node_index})")

        # ルートノードがない場合、最初のノードをルートとして使用
        if not root_nodes and len(gltf_json["nodes"]) > 0:
            if debug:
                print("Warning: No root nodes found. Using the first node as root.")
            root_nodes = [0]

        # ルートノードをシーンに設定
        gltf_json["scenes"][0]["nodes"] = root_nodes

        # ノード階層を出力（デバッグ用）
        if debug:
            print("\nGLTF Node Hierarchy:")
            for i, node in enumerate(gltf_json["nodes"]):
                children = node.get("children", [])
                children_str = ', '.join([f"{c} ({gltf_json['nodes'][c]['name']})" for c in children]) if children else "none"
                print(f"Node {i} ({node['name']}): children = [{children_str}]")

            print(f"\nRoot nodes: {root_nodes}")

        # GLBファイルの出力
        # GLBはJSON部分のみのファイルとなる
        if output_path.endswith('.glb'):
            # GLBの代わりにGLTFを出力
            gltf_path = output_path.replace('.glb', '.gltf')
        else:
            gltf_path = output_path + '.gltf'

        # GLTFファイルとして保存
        with open(gltf_path, 'w', encoding='utf-8') as f:
            json.dump(gltf_json, f, indent=2)

        if debug:
            print(f"Nodes-only GLTFファイルを生成しました: {gltf_path}")

        return {
            'glb_path': gltf_path  # 注意: 実際にはGLTFファイル
        }

    # ここから先はメッシュがある場合の処理（元の関数と同じ）
    # テクスチャが存在しないメッシュ用のダミーテクスチャ作成
    for node_name in mesh_names:
        if node_name not in texture_images:
            # ダミーテクスチャを作成して保存
            if debug:
                print(f"Info: Creating dummy texture for mesh '{node_name}'")
            dummy_texture = Image.new('RGB', (2, 2), color='white')
            texture_images[node_name] = dummy_texture

    # バッファの準備
    vertex_data = {}
    index_data = {}
    uv_data = {}
    image_data = {}

    # 各メッシュのバイナリデータを準備
    for mesh_name in mesh_names:
        # 頂点データ
        vertex_data[mesh_name] = bytearray()
        for vertex in meshes[mesh_name]['vertices']:
            vertex_data[mesh_name].extend(struct.pack('fff', *vertex))

        # インデックスデータ
        index_data[mesh_name] = bytearray()
        for face in meshes[mesh_name]['faces']:
            for idx in face:
                index_data[mesh_name].extend(struct.pack('H', idx))

        # UVデータ
        uv_data[mesh_name] = bytearray()
        for uv in meshes[mesh_name]['uvs']:
            uv_data[mesh_name].extend(struct.pack('ff', *uv))

        # 画像データ
        if mesh_name in texture_images and texture_images[mesh_name] is not None:
            image_buffer = io.BytesIO()
            texture_images[mesh_name].save(image_buffer, format="PNG")
            image_data[mesh_name] = image_buffer.getvalue()
        else:
            # ダミーテクスチャデータ (白い2x2ピクセル)
            dummy_buffer = io.BytesIO()
            dummy_texture = Image.new('RGB', (2, 2), color='white')
            dummy_texture.save(dummy_buffer, format="PNG")
            image_data[mesh_name] = dummy_buffer.getvalue()

    # バッファ全体を構築
    buffer_data = bytearray()

    # オフセット情報を保持する辞書
    offsets = {}

    # 現在のオフセット位置
    current_offset = 0

    # 各メッシュのデータを順番にバッファに追加
    for mesh_name in mesh_names:
        # 頂点データ
        offsets[f'{mesh_name}_vertex'] = {
            'offset': current_offset,
            'length': len(vertex_data[mesh_name])
        }
        buffer_data.extend(vertex_data[mesh_name])
        current_offset += len(vertex_data[mesh_name])

        # インデックスデータ
        offsets[f'{mesh_name}_index'] = {
            'offset': current_offset,
            'length': len(index_data[mesh_name])
        }
        buffer_data.extend(index_data[mesh_name])
        current_offset += len(index_data[mesh_name])

        # UVデータ
        offsets[f'{mesh_name}_uv'] = {
            'offset': current_offset,
            'length': len(uv_data[mesh_name])
        }
        buffer_data.extend(uv_data[mesh_name])
        current_offset += len(uv_data[mesh_name])

    # 画像データを最後にまとめて追加
    for mesh_name in mesh_names:
        offsets[f'{mesh_name}_image'] = {
            'offset': current_offset,
            'length': len(image_data[mesh_name])
        }
        buffer_data.extend(image_data[mesh_name])
        current_offset += len(image_data[mesh_name])

    # pygltflib構造の作成
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")

    # シーンとノード構造
    gltf.scenes.append(Scene(name="Scene"))
    gltf.scene = 0

    # シーングラフからノード階層を作成
    nodes_dict = {}  # ノード名とインデックスのマッピング
    node_index = 0

    # 有効なノード名のセットを作成 (メッシュノードとその親ノード、および構造ノード)
    valid_nodes = set(mesh_names) | structure_nodes

    # シーングラフを走査してノード構造を作成
    # 最初にすべての有効なノードを作成
    for node_name in valid_nodes:
        # メッシュノードかどうかを確認
        is_mesh_node = node_name in mesh_names
        mesh_index = mesh_names.index(node_name) if is_mesh_node else None

        # カスタム変換行列を取得
        if node_name in custom_transforms:
            transform = custom_transforms[node_name]
        else:
            transform = scene.graph[node_name][0] if node_name in scene.graph else np.eye(4)

        # 変換を分解: 平行移動、回転、スケールに
        translation = transform[:3, 3].tolist()

        # 回転は四元数に変換する必要がある
        # この例では簡略化して単位四元数を使用
        rotation = [0.0, 0.0, 0.0, 1.0]  # デフォルトの回転なし(x, y, z, w)

        # スケールは行列から抽出（簡易化）
        scale = [1.0, 1.0, 1.0]  # デフォルトのスケール

        # ノードを作成
        node = Node(
            name=node_name,
            mesh=mesh_index,
            translation=translation,
            rotation=rotation,
            scale=scale
        )

        gltf.nodes.append(node)
        nodes_dict[node_name] = node_index
        node_index += 1

    # 親子関係を設定 (カスタム階層を使用)
    for node_name, parent_name in custom_hierarchy.items():
        if parent_name is not None and parent_name in nodes_dict and node_name in nodes_dict:
            parent_index = nodes_dict[parent_name]
            child_index = nodes_dict[node_name]

            # 親ノードの子リストを更新
            if not hasattr(gltf.nodes[parent_index], 'children') or gltf.nodes[parent_index].children is None:
                gltf.nodes[parent_index].children = []

            # 既に子リストに追加されていない場合のみ追加
            if child_index not in gltf.nodes[parent_index].children:
                gltf.nodes[parent_index].children.append(child_index)
                if debug:
                    print(f"Added node '{node_name}' (index {child_index}) as child of '{parent_name}' (index {parent_index})")

    # ルートノードをシーンに追加
    # カスタム階層から親がないノードを特定
    root_nodes = []
    for node_name in valid_nodes:
        if node_name in custom_hierarchy and custom_hierarchy[node_name] is None:
            if node_name in nodes_dict:
                root_node_index = nodes_dict[node_name]
                root_nodes.append(root_node_index)
                if debug:
                    print(f"Added root node: '{node_name}' (index {root_node_index})")

    # ルートノードがない場合、最初のノードをルートとして使用
    if not root_nodes and len(gltf.nodes) > 0:
        if debug:
            print("Warning: No root nodes found. Using the first node as root.")
        root_nodes = [0]

    if debug:
        # デバッグ: ノード階層を出力
        print("\nGLTF Node Hierarchy:")
        for i, node in enumerate(gltf.nodes):
            children = getattr(node, 'children', [])
            children_str = ', '.join([f"{c} ({gltf.nodes[c].name})" for c in children]) if children else "none"
            print(f"Node {i} ({node.name}): children = [{children_str}]")

        print(f"\nRoot nodes: {root_nodes}")

    gltf.scenes[0].nodes = root_nodes

    # バッファ、バッファビュー、アクセサの設定
    gltf.buffers.append(Buffer(byteLength=len(buffer_data)))

    # バッファビュー、アクセサ、テクスチャを追跡するための辞書
    buffer_views = {}
    accessors = {}
    textures = {}
    materials = {}

    # 各メッシュのバッファビューとアクセサを作成
    for i, mesh_name in enumerate(mesh_names):
        # 頂点データのバッファビュー
        vertex_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_name}_vertex']['offset'],
            byteLength=offsets[f'{mesh_name}_vertex']['length'],
            target=ARRAY_BUFFER
        )
        gltf.bufferViews.append(vertex_buffer_view)
        buffer_views[f'{mesh_name}_vertex'] = len(gltf.bufferViews) - 1

        # インデックスデータのバッファビュー
        index_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_name}_index']['offset'],
            byteLength=offsets[f'{mesh_name}_index']['length'],
            target=ELEMENT_ARRAY_BUFFER
        )
        gltf.bufferViews.append(index_buffer_view)
        buffer_views[f'{mesh_name}_index'] = len(gltf.bufferViews) - 1

        # UVデータのバッファビュー
        uv_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_name}_uv']['offset'],
            byteLength=offsets[f'{mesh_name}_uv']['length'],
            target=ARRAY_BUFFER
        )
        gltf.bufferViews.append(uv_buffer_view)
        buffer_views[f'{mesh_name}_uv'] = len(gltf.bufferViews) - 1

        # イメージデータのバッファビュー
        image_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_name}_image']['offset'],
            byteLength=offsets[f'{mesh_name}_image']['length']
        )
        gltf.bufferViews.append(image_buffer_view)
        buffer_views[f'{mesh_name}_image'] = len(gltf.bufferViews) - 1

        # 頂点データのアクセサ
        vertices_np = meshes[mesh_name]['vertices']
        # 空の配列でのmin/maxエラーを防ぐためのチェック
        if len(vertices_np) > 0:
            min_values = vertices_np.min(axis=0).tolist()
            max_values = vertices_np.max(axis=0).tolist()
        else:
            # 空の場合はデフォルト値を使用
            min_values = [0, 0, 0]
            max_values = [0, 0, 0]

        position_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_name}_vertex'],
            componentType=FLOAT,
            count=len(vertices_np),
            type=VEC3,
            min=min_values,
            max=max_values
        )
        gltf.accessors.append(position_accessor)
        accessors[f'{mesh_name}_position'] = len(gltf.accessors) - 1

        # インデックスデータのアクセサ
        index_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_name}_index'],
            componentType=UNSIGNED_SHORT,
            count=len(meshes[mesh_name]['faces']) * 3,
            type=SCALAR
        )
        gltf.accessors.append(index_accessor)
        accessors[f'{mesh_name}_indices'] = len(gltf.accessors) - 1

        # UVデータのアクセサ
        uv_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_name}_uv'],
            componentType=FLOAT,
            count=len(meshes[mesh_name]['uvs']),
            type=VEC2
        )
        gltf.accessors.append(uv_accessor)
        accessors[f'{mesh_name}_texcoord'] = len(gltf.accessors) - 1

        # イメージ
        image = GLTFImage(
            name=f"{mesh_name}_texture",
            mimeType="image/png",
            bufferView=buffer_views[f'{mesh_name}_image']
        )
        gltf.images.append(image)

        # サンプラー（まだない場合は追加）
        if len(gltf.samplers) == 0:
            sampler = Sampler(
                magFilter=9729,  # LINEAR
                minFilter=9729,  # LINEAR
                wrapS=10497,     # REPEAT
                wrapT=10497      # REPEAT
            )
            gltf.samplers.append(sampler)

        # テクスチャ
        texture = Texture(
            sampler=0,
            source=len(gltf.images) - 1
        )
        gltf.textures.append(texture)
        textures[mesh_name] = len(gltf.textures) - 1

        # マテリアル
        material = Material(
            name=f"{mesh_name}_material",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=textures[mesh_name]),
                metallicFactor=0.0,
                roughnessFactor=1.0
            ),
            alphaMode="OPAQUE"
        )
        material.alphaCutoff = None  # alphaCutoffを明示的に削除
        gltf.materials.append(material)
        materials[mesh_name] = len(gltf.materials) - 1

        # プリミティブとメッシュ
        primitive = Primitive(
            attributes=Attributes(
                POSITION=accessors[f'{mesh_name}_position'],
                TEXCOORD_0=accessors[f'{mesh_name}_texcoord']
            ),
            indices=accessors[f'{mesh_name}_indices'],
            material=materials[mesh_name]
        )

        mesh = Mesh(
            name=mesh_name,
            primitives=[primitive]
        )

        gltf.meshes.append(mesh)

    # バイナリデータを設定
    gltf.set_binary_blob(buffer_data)

    # 実際のGLBデータ生成前にJSON構造から'alphaCutoff'を削除
    if hasattr(gltf, '_json_dict') and '_json_dict' in dir(gltf):
        if 'materials' in gltf._json_dict and len(gltf._json_dict['materials']) > 0:
            for mat in gltf._json_dict['materials']:
                if 'alphaCutoff' in mat:
                    del mat['alphaCutoff']

    # GLBとして保存
    gltf.save(output_path)
    if debug:
        print(f"GLBファイルを生成しました: {output_path}")

    return {
        'glb_path': output_path
    }
