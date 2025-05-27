import cv2
import glob

tracker = cv2.TrackerCSRT_create()

# 画像のパス一覧を取得（昇順に並べる）
images = sorted(glob.glob('untitled.png'))  # 例: images/frame_001.jpg ...

# 最初の画像でROIを選択
first_frame = cv2.imread(images[0])
bbox = cv2.selectROI("Tracking", first_frame, False)
tracker.init(first_frame, bbox)

# 画像を1枚ずつ読み込んでトラッキング
for img_path in images:
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    success, box = tracker.update(frame)
    if success:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(100) & 0xFF == 27:  # ESCキーで停止
        break

cv2.destroyAllWindows()
