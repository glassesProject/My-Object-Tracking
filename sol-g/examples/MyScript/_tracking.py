from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
#import pyautogui
import sys
import os

# Add the parent directory of 'synchronous' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.streaming.gaze_stream import GazeData
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient
from ganzin.sol_sdk.utils import find_nearest_timestamp_match
from asynchronous.overlay_gaze_on_streaming_video import main



# YOLOv8モデルのロード（初回は自動でダウンロードされます）
model = YOLO("yolov8n.pt").to('cuda')

# DeepSORT初期化
tracker = DeepSort(max_age=30)

# 動画ファイル読み込み（カメラなら 0）
cap = cv2.VideoCapture(0)
print(type(cap))

if not cap.isOpened():
    print("Webカメラが開けませんでした")
    exit()

screenshot_taken = False
window_shown_time = None


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    print(type(results))

    detections = []
    class_name = []
    #confs = []
    #i = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            cls = int(box.cls[0].cpu())
            class_name.append(model.names[cls])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))


    track_id_to_label = {}
    tracks = tracker.update_tracks(detections, frame=frame)


    for i,track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        #if 
        min_dist = float('inf')
        matched_class = "unknown"
        for det in detections:
            
            dx, dy, dw, dh = det[0]
            det_center = (dx + dw / 2, dy + dh / 2)
            track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            dist = ((det_center[0] - track_center[0]) ** 2 + (det_center[1] - track_center[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                matched_class = model.names[det[2]] if det[2] < len(model.names) else "unknown"
            print("print",matched_class)


        if not matched_class == "person":
            track_id_to_label[track_id] = matched_class
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} ,{track_id_to_label[track_id]}",(x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)            
    

        #if i >=len(detections)/3:
        #    i = 0
        #else:
        #    i += 1
        #print("ID:",track_id,"x1=",x1,"y1=", y1,"x2=", x2,"y2=", y2)
    print("ppp",type(frame))
    cv2.imshow("Tracking", frame)
    if window_shown_time is None:
        window_shown_time = time.time()

    # 3秒後にスクショを1回だけ撮る
    if not screenshot_taken and window_shown_time is not None:
        nt = time.time()
        if time.time() - window_shown_time >= 3:
            windows = pygetwindow.getWindowsWithTitle("Tracking")
            if windows:
                win = windows[0]
                win.activate()
                x, y = win.topleft
                width, height = win.size
                #screenshot = pyautogui.screenshot(region=(x, y, width, height))
                #screenshot.save(f'小津　A/media_folder/{nt}_screenshot.png')
                #screenshot_taken = True  # フラグを立てて2回目以降は撮らない
      
    
    if cv2.waitKey(1) & 0xFF == 27:  # Escキーで終了
        break


cap.release()
cv2.destroyAllWindows()
