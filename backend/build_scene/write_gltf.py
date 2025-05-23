import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import struct
import io
import json
from pygltflib import GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes
from pygltflib import Buffer, BufferView, Accessor, Material, PbrMetallicRoughness
from pygltflib import Texture, Sampler, Image as GLTFImage, TextureInfo, OcclusionTextureInfo, NormalMaterialTexture

from .constants import *
from .utils import decompose_transform_matrix


def write_gltf_json(
        output_path,
        scene,
        structure_nodes,
        dict_uuid_to_node,
        custom_transforms,
        custom_hierarchy,
        debug=True,
    ):
    # GLTF構造を構築
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
    nodes_dict = {}  # ノードuuid -> インデックスのマッピング
    node_index = 0

    for geom_uuid, geom in structure_nodes.items():
        # カスタム変換行列を取得
        node_name = geom['node_name']
        if geom_uuid in custom_transforms:
            transform = custom_transforms[geom_uuid]
        else:
            transform = scene.graph[node_name][0] if node_name in scene.graph else np.eye(4)

        # 変換を分解: 平行移動、回転、スケールに
        result = decompose_transform_matrix(transform)
        if not result['success']:
            msg = result['error']
            print(f"{node_uuid}: {msg}")
        translation = result['translation']
        rotation = result['rotation']
        scale = result['scale']

    if debug:
        print(f"{geom_uuid}:{node_name}: translation={translation}, rotation={rotation}, scale={scale}")

    # ノード情報を追加
    node_info = {
        "name": node_name,
        "translation": translation,
        "rotation": rotation,
        "scale": scale
    }

    gltf_json["nodes"].append(node_info)
    nodes_dict[geom_uuid] = node_index
    node_index += 1

    # 親子関係を設定
    for node_uuid, parent_uuid in custom_hierarchy.items():
        node = dict_uuid_to_node[node_uuid]
        if parent_uuid is None:
            continue  # 親なし (worldノード)
        parent = dict_uuid_to_node[parent_uuid]

        node_name = node.metadata.get('name', None)
        parent_name = parent.metadata.get('name', None)

        if parent_uuid in nodes_dict and node_uuid in nodes_dict:
            parent_index = nodes_dict[parent_uuid]
            child_index = nodes_dict[node_uuid]

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
    for node_uuid, node_info in structure_nodes.items():
        node_name = node_info['node_name']
        if node_uuid in custom_hierarchy and custom_hierarchy[node_uuid] is None:
            if node_uuid in nodes_dict:
                root_node_index = nodes_dict[node_uuid]
                root_nodes.append(root_node_index)
                if debug:
                    print(f"Added root node: '{node_uuid}:{node_name}' (index {root_node_index})")

    # ルートノードがない場合、最初のノードをルートとして使用
    if not root_nodes and len(gltf_json["nodes"]) > 0:
        if debug:
            print("Warning: No root nodes found. Using the first node as root.")
        root_nodes = [0]

    # ルートノードをシーンに設定
    gltf_json["scenes"][0]["nodes"] = root_nodes

    # GLTFノード階層を出力（デバッグ用）
    if debug:
        print("\nGLTF Node Hierarchy:")
        for i, node in enumerate(gltf_json["nodes"]):
            children = node.get("children", [])
            children_str = ', '.join([f"{c} ({gltf_json['nodes'][c]['name']})" for c in children]) if children else "none"
            print(f"Node {i} ({node['name']}): children = [{children_str}]")

        print(f"\nRoot nodes: {root_nodes}")

    # GLBの代わりにGLTFを出力
    if output_path.endswith('.glb'):
        gltf_path = output_path.replace('.glb', '.gltf')
    else:
        gltf_path = output_path + '.gltf'

    # GLTFファイルとして保存
    with open(gltf_path, 'w', encoding='utf-8') as f:
        json.dump(gltf_json, f, indent=2)

    if debug:
        print(f"Nodes-only GLTFファイルを生成しました: {gltf_path}")
    return gltf_path


def write_gltf_binary(
        output_path,
        scene,
        meshes,
        structure_nodes,
        dict_uuid_to_node,
        custom_transforms,
        custom_hierarchy,
        texture_images,
        debug=True,
    ):

    # Trimesh用各種バッファ
    vertex_data = {}
    index_data = {}
    uv_data = {}
    image_data = {}

    # GLTF用バッファ
    buffer_data = bytearray()
    offsets = {}  # オフセット情報を保持する辞書
    current_offset = 0  # 現在のオフセット位置

    # GLTFバッファビュー、アクセサ、テクスチャを追跡するための辞書
    buffer_views = {}
    accessors = {}
    textures = {}
    materials = {}
    mesh_indices = {}  # uuid -> mesh index の辞書

    # PBR テクスチャタイプ
    texture_types = [
        "baseColorTexture",
        "metallicRoughnessTexture",
        "normalTexture",
        "occlusionTexture",
        "emissiveTexture",
    ]

    print(f'{len(texture_images.keys())} texture_image:')
    if debug:
        for k in texture_images.keys():
            print(f'texture key: {k}')

    # 各メッシュのバイナリデータを準備
    for mesh_uuid, mesh_info in meshes.items():
        mesh_name = mesh_info['node_name']
        # 頂点データ
        vertex_data[mesh_uuid] = bytearray()
        for vertex in meshes[mesh_uuid]['vertices']:
            vertex_data[mesh_uuid].extend(struct.pack('fff', *vertex))

        # インデックスデータ
        index_data[mesh_uuid] = bytearray()
        for face in meshes[mesh_uuid]['faces']:
            for idx in face:
                index_data[mesh_uuid].extend(struct.pack('H', idx))

        # UVデータ
        uv_data[mesh_uuid] = bytearray()
        for uv in meshes[mesh_uuid]['uvs']:
            # UV座標のY成分（V成分）を反転
            # trimesh内部ではPIL.Imageを参照しているのでテクスチャの+Vが逆
            flipped_uv = [uv[0], 1.0 - uv[1]]
            uv_data[mesh_uuid].extend(struct.pack('ff', *flipped_uv))

        # 画像データ
        for texture_type in texture_types:
            k_texture = f'{mesh_uuid}_{texture_type}'
            if k_texture in texture_images and texture_images[k_texture] is not None:
                image_buffer = io.BytesIO()
                texture_images[k_texture].save(image_buffer, format="PNG")
                image_data[k_texture] = image_buffer.getvalue()
            else:
                pass
                # ダミーテクスチャデータ (白い2x2ピクセル)
                # TODO: 不要と思われるので調査
                #if debug:
                #    print(f"added dummy texture for {mesh_name}:{k_texture}")
                #dummy_buffer = io.BytesIO()
                #dummy_texture = Image.new('RGB', (2, 2), color='white')
                #dummy_texture.save(dummy_buffer, format="PNG")
                #image_data[k_texture] = dummy_buffer.getvalue()

    print(f'{len(image_data.keys())} image_data:')
    if debug:
        for k in image_data.keys():
            print(f'image_data key: {k}')

    # 各メッシュのデータを順番にバッファに追加
    for mesh_uuid, mesh_info in meshes.items():
        mesh_name = mesh_info['node_name']
        # 頂点データ
        offsets[f'{mesh_uuid}_vertex'] = {
            'offset': current_offset,
            'length': len(vertex_data[mesh_uuid])
        }
        buffer_data.extend(vertex_data[mesh_uuid])
        current_offset += len(vertex_data[mesh_uuid])

        # インデックスデータ
        offsets[f'{mesh_uuid}_index'] = {
            'offset': current_offset,
            'length': len(index_data[mesh_uuid])
        }
        buffer_data.extend(index_data[mesh_uuid])
        current_offset += len(index_data[mesh_uuid])

        # UVデータ
        offsets[f'{mesh_uuid}_uv'] = {
            'offset': current_offset,
            'length': len(uv_data[mesh_uuid])
        }
        buffer_data.extend(uv_data[mesh_uuid])
        current_offset += len(uv_data[mesh_uuid])

        # 画像データを最後にまとめて追加
        for texture_type in texture_types:
            k_texture = f'{mesh_uuid}_{texture_type}'
            if k_texture in image_data:
                offsets[k_texture] = {
                    'offset': current_offset,
                    'length': len(image_data[k_texture])
                }
                buffer_data.extend(image_data[k_texture])
                current_offset += len(image_data[k_texture])

    print(f'{len(offsets.keys())} offsets:')
    if debug:
        for k in offsets.keys():
            print(f'offset key: {k}')

    # pygltflib構造の作成
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")

    # シーンとノード構造
    gltf.scenes.append(Scene(name="Scene"))
    gltf.scene = 0

    # バッファ、バッファビュー、アクセサの設定
    gltf.buffers.append(Buffer(byteLength=len(buffer_data)))

    # 各メッシュのバッファビューとアクセサを作成
    for mesh_uuid, mesh_info in meshes.items():
        mesh_name = mesh_info['node_name']
        mesh_object = dict_uuid_to_node[mesh_uuid]

        # 頂点データのバッファビュー
        vertex_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_uuid}_vertex']['offset'],
            byteLength=offsets[f'{mesh_uuid}_vertex']['length'],
            target=ARRAY_BUFFER
        )
        gltf.bufferViews.append(vertex_buffer_view)
        buffer_views[f'{mesh_uuid}_vertex'] = len(gltf.bufferViews) - 1

        # インデックスデータのバッファビュー
        index_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_uuid}_index']['offset'],
            byteLength=offsets[f'{mesh_uuid}_index']['length'],
            target=ELEMENT_ARRAY_BUFFER
        )
        gltf.bufferViews.append(index_buffer_view)
        buffer_views[f'{mesh_uuid}_index'] = len(gltf.bufferViews) - 1

        # UVデータのバッファビュー
        uv_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_uuid}_uv']['offset'],
            byteLength=offsets[f'{mesh_uuid}_uv']['length'],
            target=ARRAY_BUFFER
        )
        gltf.bufferViews.append(uv_buffer_view)
        buffer_views[f'{mesh_uuid}_uv'] = len(gltf.bufferViews) - 1

        # イメージデータのバッファビュー
        for texture_type in texture_types:
            k_texture = f'{mesh_uuid}_{texture_type}'
            if k_texture in offsets:
                image_buffer_view = BufferView(
                    buffer=0,
                    byteOffset=offsets[k_texture]['offset'],
                    byteLength=offsets[k_texture]['length']
                )
                gltf.bufferViews.append(image_buffer_view)
                buffer_views[k_texture] = len(gltf.bufferViews) - 1

        # 頂点データのアクセサ
        vertices_np = meshes[mesh_uuid]['vertices']
        # 空の配列でのmin/maxエラーを防ぐためのチェック
        if len(vertices_np) > 0:
            min_values = vertices_np.min(axis=0).tolist()
            max_values = vertices_np.max(axis=0).tolist()
        else:
            # 空の場合はデフォルト値を使用
            min_values = [0, 0, 0]
            max_values = [0, 0, 0]

        position_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_uuid}_vertex'],
            componentType=FLOAT,
            count=len(vertices_np),
            type=VEC3,
            min=min_values,
            max=max_values
        )
        gltf.accessors.append(position_accessor)
        accessors[f'{mesh_uuid}_position'] = len(gltf.accessors) - 1

        # インデックスデータのアクセサ
        index_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_uuid}_index'],
            componentType=UNSIGNED_SHORT,
            count=len(meshes[mesh_uuid]['faces']) * 3,
            type=SCALAR
        )
        gltf.accessors.append(index_accessor)
        accessors[f'{mesh_uuid}_indices'] = len(gltf.accessors) - 1

        # UVデータのアクセサ
        uv_accessor = Accessor(
            bufferView=buffer_views[f'{mesh_uuid}_uv'],
            componentType=FLOAT,
            count=len(meshes[mesh_uuid]['uvs']),
            type=VEC2
        )
        gltf.accessors.append(uv_accessor)
        accessors[f'{mesh_uuid}_texcoord'] = len(gltf.accessors) - 1

        # イメージ
        # TODO: PBR テクスチャの複数イメージに対応
        created_texture_types = {}
        for texture_type in texture_types:
            k_texture = f'{mesh_uuid}_{texture_type}'
            if k_texture in buffer_views:
                image = GLTFImage(
                    name=k_texture,
                    mimeType="image/png",
                    bufferView=buffer_views[k_texture]
                )
                gltf.images.append(image)
                image_index = len(gltf.images) - 1

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
                # TODO: PBR テクスチャの複数イメージに対応
                texture = Texture(
                    sampler=0,
                    source=image_index
                )
                gltf.textures.append(texture)
                textures[k_texture] = len(gltf.textures) - 1
                created_texture_types[k_texture] = len(gltf.textures) - 1

        mesh_material = mesh_object.visual.material
        is_pbr = isinstance(mesh_material, trimesh.visual.material.PBRMaterial)

        # マテリアル
        # TODO: PBR Material 対応
        # TODO: PBR テクスチャの複数イメージに対応
        pbr_metallic_roughness = PbrMetallicRoughness()

        # baseColorFactor
        if hasattr(mesh_material, 'baseColorFactor') and mesh_material.baseColorFactor is not None:
            f = (mesh_material.baseColorFactor / 255.0).tolist()
            print(f'baseColorFactor {f}')
            pbr_metallic_roughness.baseColorFactor = f
        else:
            pbr_metallic_roughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]  # default value

        # metallicFactor
        if hasattr(mesh_material, 'metallicFactor') and mesh_material.metallicFactor is not None:
            pbr_metallic_roughness.metallicFactor = mesh_material.metallicFactor
        else:
            pbr_metallic_roughness.metallicFactor = 0.0  # default value

        # roughnessFactor
        if hasattr(mesh_material, 'roughnessFactor') and mesh_material.roughnessFactor is not None:
            pbr_metallic_roughness.roughnessFactor = mesh_material.roughnessFactor
        else:
            pbr_metallic_roughness.roughnessFactor = 1.0  # default value

        # PBRマテリアル各テクスチャ情報の設定
        k = f'{mesh_uuid}_baseColorTexture'
        if k in created_texture_types:
            pbr_metallic_roughness.baseColorTexture = TextureInfo(
                index=created_texture_types[k]
            )

        k = f'{mesh_uuid}_metallicRoughnessTexture'
        if k in created_texture_types:
            pbr_metallic_roughness.metallicRoughnessTexture = TextureInfo(
                index=created_texture_types[k]
            )

        gltf_material = Material(
            name=k_texture,
            pbrMetallicRoughness=pbr_metallic_roughness,
            alphaMode="OPAQUE",
            alphaCutoff=None,  # alphaCutoffを明示的に削除
        )

        # 追加のPBRプロパティ
        if is_pbr:
            # emissiveFactor
            if hasattr(mesh_material, 'emissiveFactor') and mesh_material.emissiveFactor is not None:
                gltf_material.emissiveFactor = mesh_material.emissiveFactor

            # normalTexture
            k = f'{mesh_uuid}_normalTexture'
            if k in created_texture_types:
                gltf_material.normalTexture = NormalMaterialTexture(
                    index=created_texture_types[k]
                )
                if hasattr(mesh_material, 'normalScale') and mesh_material.normalScale is not None:
                    gltf_material.normalTexture.scale = mesh_material.normalScale

            # occlusionTexture
            k = f'{mesh_uuid}_occlusionTexture'
            if k in created_texture_types:
                gltf_material.occlusionTexture = OcclusionTextureInfo(
                    index=created_texture_types[k]
                )
                if hasattr(mesh_material, 'occlusionStrength') and mesh_material.occlusionStrength is not None:
                    gltf_material.occlusionTexture.strength = mesh_material.occlusionStrength

            # emissiveTexture
            k = f'{mesh_uuid}_emissiveTexture'
            if k in created_texture_types:
                gltf_material.emissiveTexture = TextureInfo(
                    index=created_texture_types[k]
                )

            # alphaMode と alphaCutoff
            if hasattr(mesh_material, 'alphaMode') and mesh_material.alphaMode is not None:
                gltf_material.alphaMode = mesh_material.alphaMode
                if mesh_material.alphaMode == "MASK" and hasattr(mesh_material, 'alphaCutoff') and mesh_material.alphaCutoff is not None:
                    gltf_material.alphaCutoff = mesh_material.alphaCutoff
                else:
                    gltf_material.alphaCutoff = None
            else:
                gltf_material.alphaCutoff = None

        gltf.materials.append(gltf_material)
        materials[mesh_uuid] = len(gltf.materials) - 1

        # プリミティブとメッシュ
        primitive = Primitive(
            attributes=Attributes(
                POSITION=accessors[f'{mesh_uuid}_position'],
                TEXCOORD_0=accessors[f'{mesh_uuid}_texcoord']
            ),
            indices=accessors[f'{mesh_uuid}_indices'],
            material=materials[mesh_uuid]
        )

        # メッシュをgltfに追加し、そのインデックスを記録
        mesh = Mesh(
            name=mesh_name,
            primitives=[primitive]
        )
        mesh_index = len(gltf.meshes)
        gltf.meshes.append(mesh)
        mesh_indices[mesh_uuid] = mesh_index

    # シーングラフからノード階層を作成
    nodes_dict = {}  # ノード名とインデックスのマッピング
    node_index = 0

    # メッシュと構造ノードの辞書を結合する
    print(f'meshes: {meshes.keys()}')
    print(f'structure_nodes: {structure_nodes.keys()}')
    duplicate_keys = set(meshes.keys()) & set(structure_nodes.keys())
    if duplicate_keys:
        print(f"duplicate between mesh and structure node dict: {duplicate_keys}")
    valid_nodes = {**meshes, **structure_nodes}

    # シーングラフを走査してノード構造を作成
    for node_uuid, node_info in valid_nodes.items():
        node_name = node_info['node_name']
        # メッシュノードかどうかを確認
        is_mesh_node = node_uuid in meshes

        # 事前に作成したmesh_indicesを使用してメッシュインデックスを設定
        if is_mesh_node and node_uuid in mesh_indices:
            node_mesh_index = mesh_indices[node_uuid]
            if debug:
                print(f"Node '{node_name}' using mesh at index {node_mesh_index}")
        else:
            node_mesh_index = None
            if debug and is_mesh_node:
                print(f"Warning: Node '{node_name}' has no corresponding mesh in mesh_indices")

        # カスタム変換行列を取得
        if node_uuid in custom_transforms:
            transform = custom_transforms[node_uuid]
        else:
            transform = scene.graph[node_name][0] if node_name in scene.graph else np.eye(4)

        # 変換を分解: 平行移動、回転、スケールに
        result = decompose_transform_matrix(transform)
        if not result['success']:
            msg = result['error']
            print(f"{node_uuid}: {msg}")
        translation = result['translation']
        rotation = result['rotation']
        scale = result['scale']

        if debug:
            print(f"{node_uuid}:{node_name}: translation={translation}, rotation={rotation}, scale={scale}")

        # ノードを作成
        node = Node(
            name=node_name,
            mesh=node_mesh_index,
            translation=translation,
            rotation=rotation,
            scale=scale
        )

        gltf.nodes.append(node)
        nodes_dict[node_uuid] = node_index
        node_index += 1

    # 親子関係を設定
    for node_uuid, parent_uuid in custom_hierarchy.items():
        node = dict_uuid_to_node[node_uuid]
        if parent_uuid is None:
            continue  # 親なし (worldノード)
        parent = dict_uuid_to_node[parent_uuid]

        node_name = node.metadata.get('name', None)
        parent_name = parent.metadata.get('name', None)

        if parent_uuid in nodes_dict and node_uuid in nodes_dict:
            parent_index = nodes_dict[parent_uuid]
            child_index = nodes_dict[node_uuid]

            # 親ノードの子リストを更新
            if gltf.nodes[parent_index].children is None:
                gltf.nodes[parent_index].children = []

            # 既に子リストに追加されていない場合のみ追加
            if child_index not in gltf.nodes[parent_index].children:
                gltf.nodes[parent_index].children.append(child_index)
                if debug:
                    print(f"Added node '{node_name}' (index {child_index}) as child of '{parent_name}' (index {parent_index})")

    # ルートノードをシーンに追加
    # カスタム階層から親がないノードを特定
    root_nodes = []
    for node_uuid, node_info in valid_nodes.items():
        node_name = node_info['node_name']
        if node_uuid in custom_hierarchy and custom_hierarchy[node_uuid] is None:
            if node_uuid in nodes_dict:
                root_node_index = nodes_dict[node_uuid]
                root_nodes.append(root_node_index)
                if debug:
                    print(f"Added root node: '{node_uuid}:{node_name}' (index {root_node_index})")

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

    # バイナリデータを設定
    gltf.set_binary_blob(buffer_data)

    # GLBとして保存
    gltf.save(output_path)
    if debug:
        print(f"GLBファイルを生成しました: {output_path}")
    return output_path