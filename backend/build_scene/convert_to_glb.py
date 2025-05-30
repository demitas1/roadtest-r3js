import trimesh
import numpy as np

from .write_gltf import write_gltf_json, write_gltf_binary


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

    # テクスチャ画像をマテリアルから取り出し、辞書に格納する関数
    def process_texture(mesh_uuid, material, texture_type, texture_images):
        if hasattr(material, texture_type) and getattr(material, texture_type) is not None:
            texture_image = getattr(material, texture_type)
            if texture_image is not None:
                # テクスチャタイプに応じたキーでイメージを保存
                k_texture = f'{mesh_uuid}_{texture_type}'
                texture_images[k_texture] = texture_image
                if debug:
                    print(f"PBR material texture {k_texture} for mesh {mesh_uuid}")

    # シーンのメッシュを探索し、バイナリバッファに格納するデータ (vertex, face, uv, texture) をすべて抽出
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
                print(f'{node_name} {geom_uuid}: No UVs. set zeros')
                uvs = np.zeros((len(vertices), 2), dtype=np.float32)

            # テクスチャ画像の取得
            if hasattr(geom.visual, 'material'):
                material = geom.visual.material
                # PBRマテリアルか？
                is_pbr = isinstance(material, trimesh.visual.material.PBRMaterial)

                if is_pbr:
                    # PBR Material で使用されているテクスチャを取得
                    print(f"PBR Material is used in {geom_uuid}:{node_name}")
                    process_texture(geom_uuid, material, 'baseColorTexture', texture_images)
                    process_texture(geom_uuid, material, 'metallicRoughnessTexture', texture_images)
                    process_texture(geom_uuid, material, 'normalTexture', texture_images)
                    process_texture(geom_uuid, material, 'occlusionTexture', texture_images)
                    process_texture(geom_uuid, material, 'emissiveTexture', texture_images)
                else:
                    # trimeshの通常テクスチャの場合, textureプロパティからイメージを取得
                    # TODO: テクスチャ画像は GLTF の PbrMetallicRoughness.baseColorTexture にセーブされる
                    #       PBRマテリアルの場合と同じキー名 "{uuid}_baseColorTexture" にあわせる
                    if hasattr(material, 'image'):
                        image = material.image
                        texture_images[geom_uuid + "_baseColorTexture"] = image
                    elif hasattr(material, 'texture'):
                        texture = material.texture
                        if hasattr(texture, 'image'):
                            texture_images[geom_uuid + "_baseColorTexture"] = texture.image

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
        # バイナリバッファが一つもない場合GLBを作成できないのでGLTFを作成する
        if debug:
            print("No valid meshes found in the scene. Creating a nodes-only GLTF file.")

        # pygltflibを使わずに直接バッファなしの純粋なノード構造のGLTFを生成
        gltf_path = write_gltf_json(
            output_path,
            scene,
            structure_nodes,
            dict_uuid_to_node,
            custom_transforms,
            custom_hierarchy,
            debug=False,
        )

        return {
            'gltf_path': gltf_path
        }

    # バイナリバッファがある場合の処理
    gltf_path = write_gltf_binary(
        output_path,
        scene,
        meshes,
        structure_nodes,
        dict_uuid_to_node,
        custom_transforms,
        custom_hierarchy,
        texture_images,
        debug=False,
    )

    return {
        'gltf_path': gltf_path
    }