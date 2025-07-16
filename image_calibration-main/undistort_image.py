import numpy as np
import cv2

# 保存しておいたカメラパラメータを読み込む
data = np.load("camera_params.npz")
mtx = data["mtx"]
dist = data["dist"]

# 補正したい画像を読み込む（JPEGやPNGなど）
image = cv2.imread("sample.jpg")

# 歪みを除去
undistorted = cv2.undistort(image, mtx, dist, None, mtx)

# 表示して確認
cv2.imshow("Before", image)
cv2.imshow("After (Undistorted)", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
