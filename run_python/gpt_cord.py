import cv2

tracker = cv2.legacy.TrackerCSRT_create()  # ← ここが重要
video = cv2.VideoCapture("video.mp4")

ret, frame = video.read()
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

    success, box = tracker.update(frame)
    if success:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
