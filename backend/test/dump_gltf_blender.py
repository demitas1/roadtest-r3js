import bpy
import sys

# コマンドライン引数からモデルのパスを取得
model_path = sys.argv[sys.argv.index('--') + 1]

# モデルをインポート
bpy.ops.import_scene.gltf(filepath=model_path)

# モデル情報をダンプ
for obj in bpy.context.scene.objects:
    print(f"Object: {obj.name}")
    if obj.type == 'MESH':
        print(f"  Vertices: {len(obj.data.vertices)}")
        print(f"  Faces: {len(obj.data.polygons)}")
