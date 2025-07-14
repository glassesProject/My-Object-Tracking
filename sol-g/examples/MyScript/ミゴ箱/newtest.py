import cv2
import numpy as np

# ベース画像（BGR）
base_img = cv2.imread("image/base_image.png")

# オーバーレイ画像（BGRA）
overlay_img = cv2.imread("image/cursor.png", cv2.IMREAD_UNCHANGED)

# オーバーレイ位置
x, y = 50, 50

# サイズ取得
h, w = overlay_img.shape[:2]

# ベース画像から対象範囲切り出し
roi = base_img[y:y+h, x:x+w].copy()

# アルファチャンネル取得（0-255 → 0.0-1.0 に変換）
alpha = overlay_img[:, :, 3] / 255.0
alpha = alpha[:, :, np.newaxis]  # チャンネル数を合わせる

# BGR部分取得
overlay_rgb = overlay_img[:, :, :3]

# アルファブレンド計算
blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)

# 合成結果を元の画像に貼り戻す
base_img[y:y+h, x:x+w] = blended

# 表示確認
cv2.imshow("Result", base_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
