from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import asyncio
import sys
import os
import cv2
import random
import math
import time
import numpy as np
import websockets
import json

# ----------------- WebSocket Server Setup -----------------

# 接続中のクライアント（Androidアプリ）を管理するセット
connected_clients = set()

# 状態を全クライアントにブロードキャストする非同期関数
async def broadcast_status(status_data):
    if connected_clients:
        message = json.dumps(status_data)
        # asyncio.waitを使って全クライアントに並行して送信
        await asyncio.wait([client.send(message) for client in connected_clients])

# クライアントが接続したときの処理
async def handler(websocket, path):
    # 新しいクライアントをセットに追加
    connected_clients.add(websocket)
    try:
        # 接続が切れるまで待機
        await websocket.wait_closed()
    finally:
        # クライアントが切断されたらセットから削除
        connected_clients.remove(websocket)


# ----------------- Object Detection Setup -----------------
model = YOLO("yolov8n.pt").to('cuda')

tracker = DeepSort(max_age=30)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.asynchronous.async_client import (
    AsyncClient, recv_video, recv_gaze
)
from ganzin.sol_sdk.common_models import Camera


# ----------------- Global Variables -----------------
frame_skip = 5  # Nフレームに1回だけYOLO + DeepSORT実行
frame_count = 0
prev_detections = []
prev_tracks = []

distance = 3
randID = None
distance_score = None

b = 0
g = 0
r = 0

count = 0

_i_ = 0
shotFlag = False
time_flag = False
box_flag = False
class_name = "N/A" # class_nameをグローバル変数として初期化

# ----------------- Main Logic -----------------

async def main():
    address, port = get_ip_and_port()
    timeout_seconds = 5.0

    async with AsyncClient(address, port) as ac:
        error_event = asyncio.Event()

        frames = asyncio.Queue(1)
        collect_video_task = asyncio.create_task(keep_last_video_frame(ac, frames, error_event))
        
        gazes = asyncio.Queue()
        collect_gaze_task = asyncio.create_task(collect_gaze(ac, gazes, error_event, timeout_seconds))

        try:
            await draw_gaze_on_frame(frames, gazes, error_event, timeout_seconds)
        finally:
            collect_video_task.cancel()
            collect_gaze_task.cancel()

async def keep_last_video_frame(ac: AsyncClient, queue: asyncio.Queue, error_event: asyncio.Event) -> None:
    async for frame in recv_video(ac, Camera.SCENE):
        if error_event.is_set():
            break

        if queue.full():
            queue.get_nowait()
        queue.put_nowait(frame)

async def collect_gaze(ac: AsyncClient, queue: asyncio.Queue, error_event: asyncio.Event, timeout) -> None:
    try:
        async for gaze in recv_gaze(ac):
            if error_event.is_set():
                break
            await asyncio.wait_for(queue.put(gaze), timeout=timeout)
    except Exception as e:
        error_event.set()

async def draw_gaze_on_frame(frame_queue, gazes, error_event: asyncio.Event, timeout):
    global b, g, r, count, _i_, shotFlag, time_flag, box_flag, class_name
    
    start_time = 0 # start_timeを初期化

    while not error_event.is_set():
        frame = await get_video_frame(frame_queue, timeout)
        gaze = await find_gaze_near_frame(gazes, frame.get_timestamp(), timeout)
        frame_buffer = frame.get_buffer()

        center = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))
        
        if shotFlag is False:
            new_frame_buffer, newCenter, score, current_class_name = tracking(frame_buffer, center)
            class_name = current_class_name # グローバル変数を更新
        else:
            new_frame_buffer = frame_buffer
            score = 999 # shotFlagがTrueのときはスコアをリセット
            newCenter = [0,0,0,0]

        if score < 2.5:
            b = 0
            g = 0
            r = 255
            if not time_flag:
                start_time = time.time()
                time_flag = True
            
            elapsed_time = time.time() - start_time
            
            if not shotFlag and elapsed_time > 3: # 3秒以上経過したら
                # ... (画像保存のロジックは省略) ...
                _i_ += 1
                if _i_ >= 3:
                   shotFlag = True
                   print("shoted")
                   box_flag = True
                   count += 1
                   _i_ = 0

        elif score < 4.5:
            b = 255
            g = 255
            r = 0
            time_flag = False
        else:
            g = 255
            r = 255
            b = 0
            time_flag = False

        if not box_flag:
            cv2.rectangle(new_frame_buffer, (newCenter[0], newCenter[1]), (newCenter[2], newCenter[3]), (b, g, r), 2)
        
        radius = 30
        bgr_color = (255, 255, 0)
        thickness = 5
        cv2.circle(new_frame_buffer, center, radius, bgr_color, thickness)

        cv2.imshow('Press "q" to exit', new_frame_buffer)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            error_event.set()
            break
        elif key == ord('a'):
            shotFlag = False
            box_flag = False
            
        # 状態を辞書にまとめる
        status_to_send = {
            "randID": str(randID) if randID is not None else "N/A",
            "distanceScore": f"{score:.2f}",
            "className": class_name if class_name else "N/A",
            "shotFlag": shotFlag
        }
        print(f"Broadcasting status: {status_to_send}") # ★この行を追加
        # 状態をブロードキャスト
        await broadcast_status(status_to_send)

async def get_video_frame(queue, timeout):
    return await asyncio.wait_for(queue.get(), timeout=timeout)

async def find_gaze_near_frame(queue, timestamp, timeout):
    item = await asyncio.wait_for(queue.get(), timeout=timeout)
    if item.get_timestamp() >= timestamp:
        return item

    while not queue.empty():
        next_item = queue.get_nowait()
        if next_item.get_timestamp() >= timestamp:
            return next_item
        item = next_item
    return item

def tracking(frame, gazePoint):
    global frame_count, prev_detections, prev_tracks, randID, distance_score

    frame_count += 1
    run_detection = (frame_count % frame_skip == 0)

    detections = []
    if run_detection:
        results = model(frame, imgsz=320)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu())
                conf = float(box.conf[0].cpu())
                detections.append(([x1, y1, x2 - x1, y2 - y1], cls, conf))
        prev_detections = detections
    else:
        detections = prev_detections

    prev_tracks = tracker.update_tracks(detections, frame=frame)
    
    id1name = model.names
    track_id_to_label = {}
    
    for track in prev_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        
        
        
        
        min_dist = float('inf')
        matched_class = "unknown"
        
        
        ltrb = track.to_ltrb()
        track_center_x = (ltrb[0] + ltrb[2]) / 2
        track_center_y = (ltrb[1] + ltrb[3]) / 2
        track_center = (track_center_x, track_center_y)

        for det in detections:
            dx, dy, dw, dh = det[0]
            det_center_x = dx + dw * 0.5
            det_center_y = dy + dh * 0.5
            dist = (det_center_x - track_center[0])**2 + (det_center_y - track_center[1])**2
            
            if dist < min_dist:
                min_dist = dist
                matched_class = id1name[det[1]] if det[1] < len(id1name) else "unknown"
        
        if matched_class != "person":
            track_id_to_label[track_id] = matched_class
            
    confirmed_tracks = [track for track in prev_tracks if track.is_confirmed()]

    if randID is None and confirmed_tracks:
        randTrack = random.choice(confirmed_tracks)
        randID = randTrack.track_id

    idList = [track.track_id for track in confirmed_tracks]
    coordinaite = [0, 0, 0, 0]
    current_class_name = "N/A"
    
    if randID in idList:
        for track_2 in confirmed_tracks:
            if track_2.track_id == randID:
                x1, y1, x2, y2 = map(int, track_2.to_ltrb())
                coordinaite = [x1, y1, x2, y2]
                current_class_name = track_id_to_label.get(randID, "unknown")
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                distance = math.sqrt((gazePoint[0] - center_x)**2 + (gazePoint[1] - center_y)**2)
                distance_score = distance / 100.0
                break
    else:
        randID = None
        distance_score = 999

    return frame, coordinaite, distance_score, current_class_name


async def run_all():
    """WebSocketサーバーとメインの処理を同時に起動する"""
    # async withを使うと、サーバーの起動と停止を安全に管理できます
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await main()  # メインの処理を実行します

if __name__ == "__main__":
    try:
        # asyncio.run()で非同期プログラムを開始します
        # この一行でイベントループの管理が自動的に行われます
        asyncio.run(run_all())
    except KeyboardInterrupt:
        # Ctrl+Cで停止したときにメッセージを表示します
        print("Program stopped by user.")