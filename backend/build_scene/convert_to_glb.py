import trimesh
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import struct
import io
import os
import json
from pygltflib import GLTF2, Asset, Scene, Node, Mesh, Primitive, Attributes
from pygltflib import Buffer, BufferView, Accessor, Material, PbrMetallicRoughness
from pygltflib import Texture, Sampler, Image as GLTFImage, TextureInfo

from .constants import *


def decompose_transform_matrix(transform):
    # 変換を分解: 平行移動、回転、スケールに
    translation = transform[:3, 3].tolist()

    # 回転行列部分を抽出
    rotation_matrix = transform[:3, :3]

    # SVD（特異値分解）を使用して回転とスケールを分離
    try:
        # NumPyのSVD分解を使用
        U, S, Vt = np.linalg.svd(rotation_matrix)

        # 回転行列 = U * Vt (純粋な回転成分)
        pure_rotation = np.dot(U, Vt)

        # SVDの結果、特異値Sがスケール成分になる
        scale = S.tolist()

        # 行列式がマイナスの場合（左手系）の処理
        det = np.linalg.det(pure_rotation)
        if det < 0:
            # 左手系になる場合の修正（最後の列を反転）
            U[:, -1] = -U[:, -1]
            scale[-1] = -scale[-1]
            pure_rotation = np.dot(U, Vt)

        # 回転行列から四元数へ変換
        r = Rotation.from_matrix(pure_rotation)
        quat = r.as_quat()  # [x, y, z, w]の順
        rotation = quat.tolist()
        result = True
        error_msg = ""

    except Exception as e:
        # 行列分解に失敗した場合はデフォルト値を使用
        rotation = [0.0, 0.0, 0.0, 1.0]  # デフォルトの回転なし(x, y, z, w)
        scale = [1.0, 1.0, 1.0]  # デフォルトのスケール
        result = False
        error_msg = "Failed to decompose transformation matrix. Using default rotation and scale values"

    return {
        'translation':translation,
        'rotation': rotation,
        'scale': scale,
        'success': result,
        'error': error_msg,
    }


def convert_to_glb(scene, output_path="./static/output.glb", debug=True):
    """
    trimeshのシーンをGLBファイルに変換する

    Args:
        scene (trimesh.Scene): 変換するtrimeshシーン
        output_path (str): 出力するGLBファイルのパス
        debug (bool): デバッグ情報を出力するかどうか

    Returns:
        {
            'gltf_path': 保存されたGLB/GLTFファイルのパス
        }
    """
    # カスタム階層情報を取得（存在する場合）
    custom_hierarchy = scene.metadata.get('custom_hierarchy', {})
    custom_transforms = scene.metadata.get('custom_transforms', {})
    dict_uuid_to_name = scene.metadata.get('uuid_to_name', {})
    dict_uuid_to_node = scene.metadata.get('uuid_to_node', {})

    if debug:
        # scene.geometryの全ジオメトリをチェック
        print("\nScene geometries:")
        for geom_key, geometry in scene.geometry.items():
            if hasattr(geometry, 'metadata') and 'uuid' in geometry.metadata:
                geom_uuid = geometry.metadata['uuid']
                node = dict_uuid_to_node[geom_uuid]
                if node:
                    node_name = node.metadata['name']
                else:
                    node_name = None
                print(f"geometry key {geom_key}: uuid {geom_uuid}: node name: {node_name}")
            else:
                print(f"geometry key {geom_key}: (no uuid)")

        # Custom の親子関係を表示
        print("\nCustom hierarchy:")
        for node_uuid, parent_uuid in custom_hierarchy.items():
            node = dict_uuid_to_node[node_uuid]
            node_name = node.metadata.get('name', None)
            if parent_uuid is None:
                print(f"Node: {node_uuid},{node_name} -> (No parent)")
            else:
                print(f"Node: {node_uuid},{node_name} -> parent {parent_uuid}")


    # シーンからメッシュ情報を取得
    meshes = {}  # 有効な頂点リストを持つメッシュの辞書 (uuid -> Trimesh)
    texture_images = {}  # node uuid をキーとした画像データを保持する辞書
    structure_nodes = {}  # 構造ノード（メッシュがないノード）の辞書

    for geom_key, geom in scene.geometry.items():
        if geom is None:
            # 通常のtrimeshシーンでNoneとなることはないはず
            print(f"Info: Node '{geom_key}' has no geometry. skip this.")
            continue

        # ジオメトリ名はtrimeshによって変更される可能性があるので
        # uuidが一致するメッシュを探し、その名前（ユーザーによってつけられた名前）を得る
        if 'uuid' in geom.metadata:
            geom_uuid = geom.metadata['uuid']
            node_name = geom.metadata['name']
        else:
            # uuidを持っていないものがあれば付与する
            geom_uuid = str(uuid.uuid4())
            geom.metadata['uuid'] = geom_uuid
            node_name = geom_key
            geom.metadata['name'] = node_name
            dict_uuid_to_node[geom_uuid] = geom

        if isinstance(geom, trimesh.Trimesh):  # メッシュ
            # 頂点座標、面情報、UV座標を取得
            vertices = geom.vertices.astype(np.float32)
            faces = geom.faces

            # 空のメッシュの場合は構造ノードとして扱う
            if len(vertices) == 0 or len(faces) == 0:
                if debug:
                    print(f"Info: Node '{node_name}' has no geometry, treating as structure node")
                structure_nodes[geom_uuid] = {
                    'node_name': node_name,
                    'vertices': None,
                    'faces': None,
                    'uvs': None,
                    'geometry': geom,
                }
                continue

            # UV座標の取得
            if hasattr(geom.visual, 'uv') and geom.visual.uv is not None:
                uvs = geom.visual.uv.astype(np.float32)
            else:
                # UVがない場合は頂点数分のデフォルト値を作成
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)

            # テクスチャ画像の取得
            # TODO: PBR Material への対応
            if hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'image'):
                image = geom.visual.material.image
                texture_images[geom_uuid] = image
            elif hasattr(geom.visual, 'material') and hasattr(geom.visual.material, 'texture'):
                # textureプロパティからイメージを取得
                texture = geom.visual.material.texture
                if hasattr(texture, 'image'):
                    texture_images[geom_uuid] = texture.image

            # 有効な頂点を持つメッシュとして追加
            meshes[geom_uuid] = {
                'node_name': node_name,
                'vertices': vertices,
                'faces': faces,
                'uvs': uvs,
                'geometry': geom,
            }
        elif isinstance(geom, trimesh.PointCloud):
            # PointCloudオブジェクトの処理
            if debug:
                print(f"Info: Node '{node_name}' is a PointCloud, special handling")

            # GLTFでは点群を扱うことができないため、小さなマーカーなどに変換する必要がある
            # ここでは省略して構造ノードとして扱う
            structure_nodes[geom_uuid] = {
                'node_name': node_name,
                'vertices': None,
                'faces': None,
                'uvs': None,
                'geometry': geom,
            }
        elif isinstance(geom, trimesh.Path):
            # Pathオブジェクトの処理
            if debug:
                print(f"Info: Node '{node_name}' is a Path, special handling")

            # GLTFでは線を扱うことができないため、必要に応じて変換する必要がある
            # ここでは省略して構造ノードとして扱う
            structure_nodes[geom_uuid] = {
                'node_name': node_name,
                'vertices': None,
                'faces': None,
                'uvs': None,
                'geometry': geom,
            }
        else:
            # その他のジオメトリタイプ
            if debug:
                print(f"Info: Node '{node_name}' has unsupported geometry type: {type(geom)}, treating as structure node")
            structure_nodes[geom_uuid] = {
                'node_name': node_name,
                'vertices': None,
                'faces': None,
                'uvs': None,
                'geometry': geom,
            }

    if debug:
        print("\nStructure nodes (non-mesh):")
        for geom_uuid, structure_node in structure_nodes.items():
            node_name = structure_node['node_name']
            print(f"- {geom_uuid}: {node_name}")

        print("\nMesh nodes:")
        for geom_uuid, mesh_info in meshes.items():
            node_name = mesh_info['node_name']
            print(f"- {geom_uuid}: {node_name}")

    # メッシュノードが存在するか確認
    if len(meshes) == 0:
        if debug:
            print("No valid meshes found in the scene. Creating a nodes-only GLTF file.")

        # バイナリバッファが一つもない場合GLBを作成できないので
        # pygltflibを使わずに直接バッファなしの純粋なノード構造のGLTFを生成

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
            print(f"{node_uuid}:{node_name}: translation={translation}, rotation={rotation}, scale={scale}")

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

        return {
            'gltf_path': gltf_path
        }

    # 以降はバイナリバッファがある場合の処理

    # テクスチャが存在しないメッシュ用のダミーテクスチャ作成
    for node_uuid, node_info in meshes.items():
        node_name = node_info['node_name']
        if node_uuid not in texture_images:
            # ダミーテクスチャを作成して保存
            if debug:
                print(f"Info: Creating dummy texture for mesh '{node_uuid}:{node_name}'")
            dummy_texture = Image.new('RGB', (2, 2), color='white')
            texture_images[node_uuid] = dummy_texture

    # バッファの準備
    vertex_data = {}
    index_data = {}
    uv_data = {}
    image_data = {}

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
        if mesh_uuid in texture_images and texture_images[mesh_uuid] is not None:
            image_buffer = io.BytesIO()
            texture_images[mesh_uuid].save(image_buffer, format="PNG")
            image_data[mesh_uuid] = image_buffer.getvalue()
        else:
            # ダミーテクスチャデータ (白い2x2ピクセル)
            dummy_buffer = io.BytesIO()
            dummy_texture = Image.new('RGB', (2, 2), color='white')
            dummy_texture.save(dummy_buffer, format="PNG")
            image_data[mesh_uuid] = dummy_buffer.getvalue()

    # バッファ全体を構築
    buffer_data = bytearray()
    offsets = {}  # オフセット情報を保持する辞書
    current_offset = 0  # 現在のオフセット位置

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
    for mesh_uuid, mesh_info in meshes.items():
        mesh_name = mesh_info['node_name']
        offsets[f'{mesh_uuid}_image'] = {
            'offset': current_offset,
            'length': len(image_data[mesh_uuid])
        }
        buffer_data.extend(image_data[mesh_uuid])
        current_offset += len(image_data[mesh_uuid])

    # pygltflib構造の作成
    gltf = GLTF2()
    gltf.asset = Asset(version="2.0")

    # シーンとノード構造
    gltf.scenes.append(Scene(name="Scene"))
    gltf.scene = 0

    # バッファ、バッファビュー、アクセサの設定
    gltf.buffers.append(Buffer(byteLength=len(buffer_data)))

    # バッファビュー、アクセサ、テクスチャを追跡するための辞書
    buffer_views = {}
    accessors = {}
    textures = {}
    materials = {}
    mesh_indices = {}  # uuid -> mesh index の辞書

    # 各メッシュのバッファビューとアクセサを作成
    for mesh_uuid, mesh_info in meshes.items():
        mesh_name = mesh_info['node_name']
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
        image_buffer_view = BufferView(
            buffer=0,
            byteOffset=offsets[f'{mesh_uuid}_image']['offset'],
            byteLength=offsets[f'{mesh_uuid}_image']['length']
        )
        gltf.bufferViews.append(image_buffer_view)
        buffer_views[f'{mesh_uuid}_image'] = len(gltf.bufferViews) - 1

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
        image = GLTFImage(
            name=f"{mesh_uuid}_texture",
            mimeType="image/png",
            bufferView=buffer_views[f'{mesh_uuid}_image']
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
        textures[mesh_uuid] = len(gltf.textures) - 1

        # マテリアル
        material = Material(
            name=f"{mesh_uuid}_material",
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorTexture=TextureInfo(index=textures[mesh_uuid]),
                metallicFactor=0.0,
                roughnessFactor=1.0
            ),
            alphaMode="OPAQUE"
        )
        material.alphaCutoff = None  # alphaCutoffを明示的に削除
        gltf.materials.append(material)
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
        # TODO: 別モジュールにする際に結果をクラス化する
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
        'gltf_path': output_path
    }
