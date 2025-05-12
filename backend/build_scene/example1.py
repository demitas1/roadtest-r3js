import numpy as np
import os
from PIL import Image
from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Attributes, Material, 
    PbrMetallicRoughness, Texture, Image as GLTFImage, Sampler,
    Buffer, BufferView, Accessor, TextureInfo
)
import base64

from .constants import *

def create_example_scene():
    # BufferTargetの値
    ELEMENT_ARRAY_BUFFER = 34963
    ARRAY_BUFFER = 34962

    # 三角形の頂点（Z-up座標系）
    vertices = np.array([
        [0, 0, 0],  # 左下
        [1, 0, 0],  # 右下
        [0, 0, 1]   # 左上
    ], dtype=np.float32)

    # 三角形の面（インデックス）
    indices = np.array([0, 1, 2], dtype=np.uint16)

    # テクスチャ座標
    uvs = np.array([
        [0, 0],  # 左下
        [1, 0],  # 右下
        [0, 1]   # 左上
    ], dtype=np.float32)

    # X軸上の位置（-5, 0, +5）
    positions = [-5, 0, 5]

    # 画像ファイルのパス - 直接使用するパス
    image_path = "./static/TestColorGrid.png"  # 実際のファイルパスに変更してください

    # GLTF2オブジェクトを作成
    gltf = GLTF2()

    # シーンを作成
    scene = Scene()
    gltf.scenes.append(scene)
    gltf.scene = 0  # デフォルトシーンを設定

    # ルートノードを作成
    root_node = Node(name="world", children=[1, 2, 3])  # 子ノードのインデックスは後で追加
    gltf.nodes.append(root_node)

    # バイナリデータを格納するバッファを作成
    buffer_indices = Buffer(byteLength=indices.nbytes)
    buffer_vertices = Buffer(byteLength=vertices.nbytes)
    buffer_uvs = Buffer(byteLength=uvs.nbytes)

    gltf.buffers.extend([buffer_indices, buffer_vertices, buffer_uvs])

    # バッファビューを作成
    buffer_view_indices = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=indices.nbytes,
        target=ELEMENT_ARRAY_BUFFER
    )

    buffer_view_vertices = BufferView(
        buffer=1,
        byteOffset=0,
        byteLength=vertices.nbytes,
        target=ARRAY_BUFFER
    )

    buffer_view_uvs = BufferView(
        buffer=2,
        byteOffset=0,
        byteLength=uvs.nbytes,
        target=ARRAY_BUFFER
    )

    gltf.bufferViews.extend([buffer_view_indices, buffer_view_vertices, buffer_view_uvs])

    # アクセサを作成
    accessor_indices = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=UNSIGNED_SHORT,
        count=len(indices),
        type=SCALAR,
        max=[int(indices.max())],
        min=[int(indices.min())]
    )

    accessor_vertices = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=FLOAT,
        count=len(vertices),
        type=VEC3,
        max=vertices.max(axis=0).tolist(),
        min=vertices.min(axis=0).tolist()
    )

    accessor_uvs = Accessor(
        bufferView=2,
        byteOffset=0,
        componentType=FLOAT,
        count=len(uvs),
        type=VEC2,
        max=uvs.max(axis=0).tolist(),
        min=uvs.min(axis=0).tolist()
    )

    gltf.accessors.extend([accessor_indices, accessor_vertices, accessor_uvs])

    # テクスチャ用の画像を設定
    if os.path.exists(image_path):
        # 画像を直接使用（一時変換なし）
        print(f"画像ファイルを直接使用します: {image_path}")

        # 画像を追加
        gltf_image = GLTFImage(uri=image_path, name="texture")
        gltf.images = [gltf_image]

        # サンプラーを追加
        sampler = Sampler(
            magFilter=9729,  # LINEAR
            minFilter=9729,  # LINEAR
            wrapS=10497,     # REPEAT
            wrapT=10497      # REPEAT
        )
        gltf.samplers = [sampler]

        # テクスチャを追加
        texture = Texture(source=0, sampler=0, name="texture")
        gltf.textures = [texture]

        has_texture = True

        # ファイルの存在を確認し、詳細を表示
        try:
            img = Image.open(image_path)
            print(f"画像サイズ: {img.size}, フォーマット: {img.format}, モード: {img.mode}")
            img.close()
        except Exception as e:
            print(f"画像情報の取得に失敗: {e}")
    else:
        print(f"警告: 画像ファイル '{image_path}' が見つかりません。テクスチャなしで続行します。")
        has_texture = False

    # 各三角形に共有マテリアルを作成（テクスチャのみ）
    pbr = PbrMetallicRoughness(
        # 完全な白色で設定してテクスチャの色が100%表示されるようにする
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,      # メタリックはゼロ
        roughnessFactor=0.5      # ラフネスは中間値
    )

    if has_texture:
        pbr.baseColorTexture = TextureInfo(index=0)

    material = Material(
        name="texture_material",
        pbrMetallicRoughness=pbr,
        doubleSided=True  # 両面描画を有効に
    )

    gltf.materials = [material]

    # 各三角形用のメッシュを作成（すべて同じマテリアルを使用）
    gltf.meshes = []
    for i in range(3):
        primitive = Primitive(
            attributes=Attributes(POSITION=1, TEXCOORD_0=2),
            indices=0,
            material=0,  # 全メッシュで同じマテリアル（index=0）を使用
            mode=4       # TRIANGLES
        )

        mesh = Mesh(
            primitives=[primitive],
            name=f"triangle_{i}"
        )

        gltf.meshes.append(mesh)

    # 各三角形のノードを作成
    for i, position in enumerate(positions):
        # 変換行列を作成
        if position != 0:
            # 位置が0でない場合は変換行列を設定
            matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                position, 0.0, 0.0, 1.0
            ]
            node = Node(mesh=i, matrix=matrix, name=f"triangle_{i}")
        else:
            # 位置が0の場合は変換行列なし
            node = Node(mesh=i, name=f"triangle_{i}")

        gltf.nodes.append(node)

    # バッファにバイナリデータを設定
    # インデックスデータをバイト列に変換
    indices_bytes = indices.tobytes()
    gltf.buffers[0].uri = f"data:application/octet-stream;base64,{base64.b64encode(indices_bytes).decode('ascii')}"

    # 頂点データをバイト列に変換
    vertices_bytes = vertices.tobytes()
    gltf.buffers[1].uri = f"data:application/octet-stream;base64,{base64.b64encode(vertices_bytes).decode('ascii')}"

    # UV座標データをバイト列に変換
    uvs_bytes = uvs.tobytes()
    gltf.buffers[2].uri = f"data:application/octet-stream;base64,{base64.b64encode(uvs_bytes).decode('ascii')}"

    # デバッグ情報を表示
    print("\nGLTF情報:")
    print(f"シーン数: {len(gltf.scenes)}")
    print(f"ノード数: {len(gltf.nodes)}")
    print(f"メッシュ数: {len(gltf.meshes)}")
    print(f"マテリアル数: {len(gltf.materials)}")
    print(f"テクスチャ数: {len(gltf.textures) if hasattr(gltf, 'textures') and gltf.textures else 0}")
    print(f"画像数: {len(gltf.images) if hasattr(gltf, 'images') and gltf.images else 0}")
    print(f"アクセサ数: {len(gltf.accessors)}")
    print(f"バッファビュー数: {len(gltf.bufferViews)}")
    print(f"バッファ数: {len(gltf.buffers)}")

    if hasattr(gltf, 'images') and gltf.images:
        print(f"画像URI: {gltf.images[0].uri}")
    if hasattr(gltf, 'materials') and gltf.materials:
        pbr = gltf.materials[0].pbrMetallicRoughness
        print(f"マテリアルのbaseColorFactor: {pbr.baseColorFactor}")
        print(f"マテリアルのテクスチャ設定: {hasattr(pbr, 'baseColorTexture') and pbr.baseColorTexture is not None}")

    # 最終的なGLTFを保存
    gltf_path = "./static/textured_triangles.gltf"
    gltf.save(gltf_path)
    print(f"\nGLTFファイルを保存しました: {gltf_path}")

    # GLB形式でも保存
    glb_path = "./static/textured_triangles.glb"
    gltf.save_binary(glb_path)
    print(f"GLBファイルを保存しました: {glb_path}")
