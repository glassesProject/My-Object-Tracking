import cv2
import numpy as np

# 歪み補正に使うカメラパラメータを読み込む
data = np.load("camera_params.npz")
mtx = data["mtx"]
dist = data["dist"]

# 新しい動画ファイルを開く
cap = cv2.VideoCapture("test.mp4")  # ← あなたの動画ファイル名に変更

# 出力動画の設定（オプション）
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("undistorted_output.mp4", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 歪み補正を適用
    undistorted = cv2.undistort(frame, mtx, dist, None, mtx)

    # 表示（確認用）
    cv2.imshow("Undistorted", undistorted)

    # 書き出し（オプション）
    out.write(undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後処理
cap.release()
out.release()
cv2.destroyAllWindows()
