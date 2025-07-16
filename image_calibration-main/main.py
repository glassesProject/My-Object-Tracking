import cv2
import numpy as np

# チェスボードの交点数（例：9x6）
chessboard_size = (10, 7)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

cap = cv2.VideoCapture("sceneview.mp4")
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if found:
        print(f"チェスボード検出成功（フレーム {frame_count}）")
        objpoints.append(objp)
        imgpoints.append(corners)

        # 検出結果を描画（確認用）
        cv2.drawChessboardCorners(frame, chessboard_size, corners, found)
        cv2.imshow('Corners', frame)
        cv2.waitKey(100)  # 少しだけ表示

    # 一定枚数だけ使う（例：15枚）
    if len(imgpoints) >= 15:
        break

cap.release()
cv2.destroyAllWindows()

if len(imgpoints) == 0:
    print("有効なフレームが見つかりませんでした。")
    exit()

# キャリブレーション実行
image_size = gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

print("カメラ行列:\n", mtx)
print("歪み係数:\n", dist)

# 保存したり補正に使ったりできます
