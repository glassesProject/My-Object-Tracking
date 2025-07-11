from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import asyncio
import sys
import os
import cv2
import random

model = YOLO("yolov8n.pt").to('cuda')

tracker = DeepSort(max_age=30)
    #embedder="mobilenet",
    #half=True,
    #bgr=True,
    #embedder_gpu=True
    #)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.asynchronous.async_client import (
    AsyncClient, recv_video, recv_gaze
)
from ganzin.sol_sdk.common_models import Camera


async def main():
    address, port = get_ip_and_port()
    timeout_seconds = 5.0

    

    async with AsyncClient(address, port) as ac:
        error_event = asyncio.Event()

        frames = asyncio.Queue(1)
        collect_video_task = asyncio.create_task(keep_last_video_frame(ac, frames, error_event))
        
        gazes = asyncio.Queue()       
        collect_gaze_task = asyncio.create_task(collect_gaze(ac, gazes, error_event, timeout_seconds))

       # return collect_video_task

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
    while not error_event.is_set():
        frame = await get_video_frame(frame_queue, timeout)
        gaze = await find_gaze_near_frame(gazes, frame.get_timestamp(), timeout)
        frame_buffer = frame.get_buffer()

        center = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))

        
        
        #new_frame_buffer = frame_buffer
        
        new_frame_buffer,newCenter = tracking(frame_buffer , center)
        cv2.rectangle(new_frame_buffer, (newCenter[0], newCenter[1]), (newCenter[2], newCenter[3]), (255, 255, 0), 2)
       # print(center)

        radius = 30
        bgr_color = (255, 255, 0)
        thickness = 5
        cv2.circle(new_frame_buffer, center, radius, bgr_color, thickness)

        cv2.imshow('Press "q" to exit', new_frame_buffer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

async def get_video_frame(queue, timeout):
    return await asyncio.wait_for(queue.get(), timeout=timeout)

async def find_gaze_near_frame(queue, timestamp, timeout):
    item = await asyncio.wait_for(queue.get(), timeout=timeout)
    if item.get_timestamp() > timestamp:
        return item

    while True:
        if queue.empty():
            return item
        else:
            next_item = queue.get_nowait()
            if next_item.get_timestamp() > timestamp:
                return next_item
            item = next_item


# 事前に定義（グローバル）
frame_skip = 5  # Nフレームに1回だけYOLO + DeepSORT実行
frame_count = 0
prev_detections = []
prev_tracks = []

distance = 3
randID = None
def tracking(frame , gazePoint):
    global frame_count, prev_detections, prev_tracks, randID

    frame_count += 1
    run_detection = (frame_count % frame_skip == 0)

    if run_detection:
        results = model(frame, imgsz=320)  # imgszでさらに高速化

        id1name = model.names
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu())
                conf = float(box.conf[0].cpu())
                detections.append(([x1, y1, x2 - x1, y2 - y1], cls, conf))

        prev_detections = detections
        prev_tracks = tracker.update_tracks(detections, frame=frame)
    else:
         #推論スキップ中は前回のトラッカー結果のみ使用
        detections = prev_detections
    id1name = model.names
    track_id_to_label = {}

    for track in prev_tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        # 最も近いクラスを探す
        min_dist = float('inf')
        matched_class = "unknown"
        for det in detections:
            dx, dy, dw, dh = det[0]
            dist_x = (dx + dw * 0.5) - track_center[0]
            dist_y = (dy + dh * 0.5) - track_center[1]
            dist = dist_x * dist_x + dist_y * dist_y
            if dist < min_dist:
                min_dist = dist
                matched_class = id1name[det[1]] if det[1] < len(id1name) else "unknown"
                
        

        if matched_class != "person":
            track_id_to_label[track_id] = matched_class
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"ID: {track_id} ,{track_id_to_label[track_id]}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if len(detections) > 5:
            for _ in range(len(detections)-5):
                del detections[random.randint(0, (len(detections)-1))]



#######---------########


    confirmed_tracks = [track for track in prev_tracks if track.is_confirmed()]

    if randID is None:
        if confirmed_tracks:
            randTrack = random.choice(confirmed_tracks)
            #print(rand_track)
            randID = randTrack.track_id
            print(f"選ばれたID:{randID}")

    idList = [track.track_id for track in confirmed_tracks]

    if randID in idList:
        for track_2 in confirmed_tracks:
            if track_2.track_id == randID:
                print(f"抽選済みID:{randID},座標:{track_2.to_ltrb()}")
                x1, x2, y1, y2 = map(int , track_2.to_ltrb())
                coordinaite = [x1, x2, y1, y2]
    else:
        randID = None
        coordinaite = [0,0,0,0]

    return frame , coordinaite


if __name__ == "__main__":
    asyncio.run(main())