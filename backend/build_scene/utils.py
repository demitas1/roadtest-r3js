import numpy as np
from scipy.spatial.transform import Rotation


def compose_transform_matrix(translation=None, rotation=None, scale=None):
    """
    移動・回転・スケールを合成して4x4変換行列を生成する
    
    Args:
        translation: list or array, 平行移動 (dx, dy, dz). デフォルト: [0, 0, 0]
        rotation: list or array, 回転成分の四元数 (x, y, z, w). デフォルト: [0, 0, 0, 1]
        scale: list or array, 拡大率 (sx, sy, sz). デフォルト: [1, 1, 1]
    
    Returns:
        numpy.ndarray: 4x4の変換行列
    """
    # デフォルト値の設定
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    if rotation is None:
        rotation = [0.0, 0.0, 0.0, 1.0]  # 回転なし
    if scale is None:
        scale = [1.0, 1.0, 1.0]  # スケールなし

    # NumPy配列に変換
    translation = np.array(translation, dtype=float)
    rotation = np.array(rotation, dtype=float)
    scale = np.array(scale, dtype=float)

    # 4x4の単位行列から開始
    transform = np.eye(4, dtype=float)

    # 1. スケール行列を作成
    scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])

    # 2. 回転行列を作成（四元数から）
    try:
        # 四元数を正規化
        quat_norm = np.linalg.norm(rotation)
        if quat_norm > 0:
            rotation = rotation / quat_norm
        else:
            rotation = np.array([0.0, 0.0, 0.0, 1.0])  # デフォルトに戻す

        # 四元数から回転行列を生成
        r = Rotation.from_quat(rotation)  # [x, y, z, w]の順
        rotation_matrix = r.as_matrix()

        # 4x4回転行列を作成
        rotation_4x4 = np.eye(4, dtype=float)
        rotation_4x4[:3, :3] = rotation_matrix

    except Exception:
        # 回転行列の生成に失敗した場合は単位行列を使用
        rotation_4x4 = np.eye(4, dtype=float)

    # 3. 平行移動を設定
    transform[:3, 3] = translation

    # 4. 変換の合成: T * R * S の順序で適用
    # まずスケール、次に回転、最後に平行移動
    transform[:3, :3] = np.dot(rotation_4x4[:3, :3], scale_matrix[:3, :3])

    return transform


def decompose_transform_matrix(transform):
    """
    変換行列を分解: 平行移動、回転、スケールに分解する

    transform: numpy.ndarray, 4x4の変換行列

    Returns:
        {
            'translation':  平行移動 (dx, dy, dz)
            'rotation':     回転成分の四元数 (x, y, z, w)
            'scale':        拡大率 (sx, sy, sz)
            'success': bool: True: 分解成功, False: 分解失敗
            'error':  str: エラーメッセージ
        }
    """
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

