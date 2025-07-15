import asyncio
import sys
import os

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Add the parent directory of 'synchronous' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.asynchronous.async_client import (
    AsyncClient, recv_video, recv_gaze
)
from ganzin.sol_sdk.common_models import Camera
import cv2

#YOLOの読み込み
model = YOLO("yolov8n.pt").to('cuda')

# DeepSORT初期化
tracker = DeepSort(max_age=30)

print(tracker)

detections = []
class_name = []

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
        print("position = " , center)
        print("adress = ",get_ip_and_port())
        radius = 30
        bgr_color = (255, 255, 0)
        thickness = 5
        #cv2.circle(frame_buffer, center, radius, bgr_color, thickness)
        #ここにオブとら

        #frame_buffer をYOLOに読み込ませる
        results = model(frame_buffer)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                bbox = [x1, y1, x2 - x1, y2 - y1]
                detections.append((bbox, conf, cls))

        track_id_to_label = {}



        tracks = tracker.update_tracks(detections, frame=frame)
        

            # for i,track in enumerate(tracks):
            # if not track.is_confirmed():
            #     continue
            # track_id = track.track_id
            # ltrb = track.to_ltrb()
            # x1, y1, x2, y2 = map(int, ltrb)
            # #if 
            # min_dist = float('inf')
            # matched_class = "unknown"
            # for det in detections:
                
            #     dx, dy, dw, dh = det[0]
            #     det_center = (dx + dw / 2, dy + dh / 2)
            #     track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            #     dist = ((det_center[0] - track_center[0]) ** 2 + (det_center[1] - track_center[1]) ** 2) ** 0.5
            #     if dist < min_dist:
            #         min_dist = dist
            #         matched_class = model.names[det[2]] if det[2] < len(model.names) else "unknown"


        cv2.imshow('Press "q" to exit', frame_buffer)
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

if __name__ == "__main__":
    asyncio.run(main())
