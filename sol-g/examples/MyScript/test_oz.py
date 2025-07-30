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
import pygame
import threading
#新聞紙作成スクリプト
import combinateimage
import printor
# from PIL import Image, ImageDraw

# データ記録機能のためにライブラリをインポート
import csv
import datetime

model = YOLO("yolov8n.pt").to('cuda')

tracker = DeepSort(max_age=15,
    n_init=2,
    nn_budget=100,
    max_cosine_distance=0.4,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_gpu=True
    )

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.asynchronous.async_client import (
    AsyncClient, recv_video, recv_gaze
)
from ganzin.sol_sdk.common_models import Camera


# 事前に定義（グローバル）
frame_skip = 3 # Nフレームに1回だけYOLO + DeepSORT実行
frame_count = 0
prev_detections = []
prev_tracks = []
remove_id = []
# グローバル制御変数
running = True

loop_channel = None  # mode=4で使うループ用チャンネル

distance = 3
randID = None
distance_score = None
score = None

b = 0
g = 0
r = 0

count = 0
#音声のモードフラグ１－４
current_mode = 1

_i_ = 0
shotFlag = False
time_flag = False
box_flag = False

async def main():
    address, port = get_ip_and_port()
    timeout_seconds = 5.0

    # === データ記録機能：実行ごとにユニークなファイル名を生成 ===
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_filename = f'interaction_log_{timestamp}.csv'
    print(f"今回のログは {log_filename} に保存されます。")
    # === ここまで ===

    async with AsyncClient(address, port) as ac:
        error_event = asyncio.Event()

        frames = asyncio.Queue(1)
        collect_video_task = asyncio.create_task(keep_last_video_frame(ac, frames, error_event))
        
        gazes = asyncio.Queue()      
        collect_gaze_task = asyncio.create_task(collect_gaze(ac, gazes, error_event, timeout_seconds))

        try:
            # ファイル名を渡す
            await draw_gaze_on_frame(frames, gazes, error_event, timeout_seconds, log_filename)
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

# log_filename を引数として受け取るように変更
async def draw_gaze_on_frame(frame_queue, gazes, error_event: asyncio.Event, timeout, log_filename):
    global b ,g ,r , count , _i_ , shotFlag , time_flag , box_flag,current_mode, remove_id,randID
    imagepaths = []

    # === データ記録機能：ヘッダーに 'session_count' を追加 ===
    csv_header = [
        'timestamp', 'frame_count', 'session_count', 'gaze_x', 'gaze_y', 
        'target_track_id', 'target_class_name', 
        'target_coord_x1', 'target_coord_y1', 'target_coord_x2', 'target_coord_y2',
        'distance_score', 'gazing_time'
    ]
    # === ここまで ===

    elapsed_time = 0

    # === データ記録機能：ファイルを開く ===
    # 'w' (上書きモード) でファイルを開き、ヘッダーを書き込む
    with open(log_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)
        # === ここまで ===

        try:
            while not error_event.is_set():
                frame = await get_video_frame(frame_queue, timeout)
                gaze = await find_gaze_near_frame(gazes, frame.get_timestamp(), timeout)

                frame_buffer = await undistort(frame.get_buffer())
                center = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))

                if shotFlag is False:
                    new_frame_buffer,newCenter,score,className = tracking(frame_buffer , center)
                else:
                    new_frame_buffer = frame_buffer
                    newCenter, score, className = [0,0,0,0], 999, None

                if score < 1.5 and  shotFlag is False :
                    set_mode(4)
                    b = 0
                    g = 0
                    r = 255
                    if time_flag is False:
                        start_time = time.time()
                        print("startTime = ",start_time)
                        time_flag = True
                    
                    if time_flag is True:
                        end_time = time.time()
                        elapsed_time = start_time - end_time
                        print("注視時間 = ",elapsed_time)

                    if elapsed_time < -3:
                        set_mode(1)
                        await play_shattersound("shattersound.wav")
                        save_dir = f"Directory_No{count}"
                        if _i_ < 3:
                            os.makedirs(save_dir, exist_ok=True)
                            file_path = os.path.join(save_dir, f"no{count}_,{_i_}.png")
                            
                            cv2.imwrite(f"rawData/No{count}_{_i_}.png", new_frame_buffer)
                            image_create(file_path,newCenter,className)
                            imagepaths.append(file_path)

                            if _i_ == 2:
                                remove_id.append(randID)
                                combinateimage.overlay_images_on_newspaper(imagepaths[0],imagepaths[1],imagepaths[2])
                                imagepaths.clear()
                                printor.print_png()

                            _i_ += 1
                            shotFlag = True
                            print("shoted")
                            box_flag = True
                        
                        else:
                            remove_id = []
                            count += 1
                            _i_ = 0

                elif score < 3 :
                    set_mode(5)
                    b = 255
                    time_flag = False
                    elapsed_time = 0
                    r = 255
                    g = 0
                elif score < 5.5 :
                    set_mode(3)
                    b = 255
                    time_flag = False
                    elapsed_time = 0
                    r = 0
                    g = 0
                elif score < 8 :
                    set_mode(2)
                    b = 0
                    time_flag = False
                    elapsed_time = 0
                    r = 0
                    g = 255
                else :
                    set_mode(1)
                    time_flag = False
                    elapsed_time = 0
                    g = 255
                    r = 255
                    b = 0

                if shotFlag :
                    current_mode = 1

                if box_flag is False:
                    cv2.rectangle(new_frame_buffer, (newCenter[0], newCenter[1]), (newCenter[2], newCenter[3]), (b, g, r), 2)
                    cv2.putText(new_frame_buffer, f"ID:{className}",(newCenter[0], newCenter[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                radius = 30
                bgr_color = (255, 255, 0)
                thickness = 5
                cv2.circle(new_frame_buffer, center, radius, bgr_color, thickness)
                
                # === データ記録機能：CSVファイルへの書き込み ===
                log_data = [
                    frame.get_timestamp(),
                    frame_count,
                    count, # session_count を記録
                    center[0],
                    center[1],
                    randID,
                    className,
                    newCenter[0],
                    newCenter[1],
                    newCenter[2],
                    newCenter[3],
                    score,
                    elapsed_time
                ]
                csv_writer.writerow(log_data)
                # === ここまで ===

                cv2.imshow('Press "q" to exit', new_frame_buffer)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('a'):
                    randID = None
                    shotFlag = False
                    box_flag = False
        except Exception as e:
            print(f"An error occurred in draw_gaze_on_frame: {e}")
        finally:
            print("draw_gaze_on_frame loop finished.")


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

def image_create(filpath , nwcen , cls):
    base_img = cv2.imread(f"rawData/No{count}_{_i_}.png")
    try:
        overlay_img = cv2.imread(f"tsukumo_image/{cls}.png", cv2.IMREAD_UNCHANGED)
    except:
        print(cls)
        othre = ["ピラミッドの付喪神","牛乳パックの付喪神","320サイズ椅子の付喪神","bottle","cell phone","keyboard","mouse","tv"]
        othre_name = random.choice(othre)
        overlay_img = cv2.imread(f"tsukumo_image/{othre_name}.png", cv2.IMREAD_UNCHANGED)

    x, y = (nwcen[0]+nwcen[2])//2, (nwcen[1]+nwcen[3])//2
    h, w = overlay_img.shape[:2]
    lux = x - w//2
    luy = y - h//2
    rbx = lux + w 
    rby = luy + h 

    if luy + h > base_img.shape[0] or lux + w > base_img.shape[1] or luy<0 or lux <0:
        print("範囲外です")
    else:
        roi = base_img[luy:rby, lux:rbx].copy()
        overlay_rgb = overlay_img[:, :, :3]
        alpha = overlay_img[:, :, 3] / 255.0
        alpha = alpha[:, :, np.newaxis]
        if roi.shape != overlay_rgb.shape:
            print("サイズ違い検出: roi.shape =", roi.shape, " overlay_rgb.shape =", overlay_rgb.shape)
            roi = cv2.resize(roi, (overlay_rgb.shape[1], overlay_rgb.shape[0]))
        blended = (overlay_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
        base_img[luy:rby, lux:rbx] = blended
        cv2.imwrite(filpath, base_img)
        cv2.imwrite('static/images/generated_image.png', base_img)

def tracking(frame , gazePoint):
    global frame_count, prev_detections, prev_tracks, randID , distance_score, remove_id

    frame_count += 1
    run_detection = (frame_count % frame_skip == 0)

    if run_detection:
        results = model(frame, imgsz=320)
        
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
        detections = prev_detections
    
    id1name = model.names
    
    for track in prev_tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
        min_dist = float('inf')
        for det in detections :
            if det[1] < len(id1name) and id1name[det[1]] not in ['person'] and id1name[det[1]] not in ['unknown']:
                dx, dy, dw, dh = det[0]
                dist_x = (dx + dw * 0.5) - track_center[0]
                dist_y = (dy + dh * 0.5) - track_center[1]
                dist = dist_x * dist_x + dist_y * dist_y
                
                if dist < min_dist:
                    min_dist = dist
                    setattr(track, "class_id", det[1])
    
    confirmed_tracks = [track for track in prev_tracks if track.is_confirmed()  and track.track_id not in remove_id]

    if randID is None :
        if confirmed_tracks:
            sound = pygame.mixer.Sound("changesound.wav")
            sound.play()
            randTrack = random.choice(confirmed_tracks)
            randID = randTrack.track_id
            distance_score = 0
            print(f"選ばれたID:{randID}")

    idList = [track.track_id for track in confirmed_tracks]
    class_name = None
    
    if idList == []:
        randID = None
        distance_score = 100
        coordinaite = [0,0,0,0]
        print("debug",randID)
    
    print("aaaa",idList)

    coordinaite = [0,0,0,0]

    if randID in idList:
        for track_2 in confirmed_tracks:
            if track_2.track_id == randID:
                class_id = getattr(track_2,"class_id",None)
                x1, y1, x2, y2 = map(int , track_2.to_ltrb())
                if x1 < 0 : x1 = 0
                if y1 < 0 : y1 = 0
                if x2 > frame.shape[1] : x2 = frame.shape[1]
                if y2 > frame.shape[0]: y2 = frame.shape[0]
                print("ボックスの座標" , x1,":",y1,":",x2,":",y2)
                coordinaite = [x1, y1, x2, y2]
                if class_id is not None and class_id < len(id1name):
                    class_name = id1name[class_id]
                else:
                    class_name = "unknown"
    else:
        randID = None
        coordinaite = [0,0,0,0]

    distance_score = 999
    if randID is not None:
        center_x = (coordinaite[0] + coordinaite[2]) / 2
        center_y = (coordinaite[1] + coordinaite[3]) / 2
        distance = math.sqrt((gazePoint[0] - center_x) ** 2 + (gazePoint[1] - center_y) ** 2)
        distance_score = distance / 100.0
        print(class_name)

    return frame , coordinaite , distance_score , class_name

async def undistort(cap):
    data = np.load(os.path.join(BASE_DIR, "camera_params.npz"))
    mtx = data["mtx"]
    dist = data["dist"]
    undistorted = cv2.undistort(cap, mtx, dist, None, mtx)
    return undistorted

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "beep.wav")
pygame.mixer.init()
beep = pygame.mixer.Sound(sound_path)

async def play_shattersound(filename):
    try:
        sound = pygame.mixer.Sound(os.path.join(BASE_DIR, filename))
        sound.play()
    except pygame.error as e:
        print(f"エラー: {e}")

# threading, time, pygame は既にインポート済み
current_mode = 1
loop_channel = None
running = True
mode_event = threading.Event()

def set_mode(new_mode):
    global current_mode
    current_mode = new_mode
    mode_event.set()
    mode_event.clear()

def beep_loop():
    global loop_channel, current_mode

    while running:
        print("音声ループ中", current_mode)
        if current_mode == 1:
            if loop_channel:
                loop_channel.stop()
                loop_channel = None
            mode_event.wait(timeout=0.1)
            continue
        elif current_mode == 2:
            interval = 1
        elif current_mode == 3:
            interval = 0.5
        elif current_mode == 5:
            interval = 0.25
        elif current_mode == 4:
            if not loop_channel or not loop_channel.get_busy():
                loop_channel = beep.play(loops=-1)
            mode_event.wait(timeout=0.1)
            continue
        else:
            interval = 1.0

        if loop_channel:
            loop_channel.stop()
            loop_channel = None
        
        channel = beep.play()
        while channel.get_busy():
            mode_event.wait(timeout=0.01)

        elapsed = 0.0
        check_interval = 0.01
        while elapsed < interval:
            if current_mode == 4:
                break
            mode_event.wait(timeout=check_interval)
            elapsed += check_interval

def start_beep_thread():
    thread = threading.Thread(target=beep_loop)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    start_beep_thread()
    asyncio.run(main())