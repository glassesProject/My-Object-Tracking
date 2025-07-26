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

model = YOLO("yolov8n.pt").to('cuda')

tracker = DeepSort(max_age=15,
    n_init=3,
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
#音声のモードフラグ１－４
current_mode = 1

_i_ = 0
shotFlag = False
time_flag = False
box_flag = False

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
    global b ,g ,r , count , _i_ , shotFlag , time_flag , box_flag,current_mode

    
    
    while not error_event.is_set():
        frame = await get_video_frame(frame_queue, timeout)
        gaze = await find_gaze_near_frame(gazes, frame.get_timestamp(), timeout)

        

        frame_buffer = await undistort(frame.get_buffer())
        #frame_buffer = frame.get_buffer()

        center = (int(gaze.combined.gaze_2d.x), int(gaze.combined.gaze_2d.y))

        
        
        #new_frame_buffer = frame_buffer
        if shotFlag is False:
            new_frame_buffer,newCenter,score,className = tracking(frame_buffer , center)
        else:
            new_frame_buffer = frame_buffer

    ###

        if score < 2.5 :
            #４は連続音
            current_mode = 4
            b = 0
            g = 0
            r = 255
            if time_flag is False:
                start_time = time.time()
                print("startTime = ",start_time)
                time_flag = True
            

            if time_flag is True:
                end_time = time.time()
                #print("endTime = ",end_time)
                elapsed_time = start_time - end_time
                print("elapsedTime = ",elapsed_time)

            if shotFlag is False and elapsed_time < -3:
                save_dir = f"Directory_No{count}"
                if _i_ < 3:
                    os.makedirs(save_dir, exist_ok=True)  # フォルダが無ければ作成

                    # 保存するファイルパス
                    file_path = os.path.join(save_dir, f"no{count}_,{_i_}.png")

                    # 画像として保存
                    cv2.imwrite(f"rawData/No{count}_{_i_}.png", new_frame_buffer)



###################################################


                    #if className == "tv":
                    base_img = cv2.imread(f"rawData/No{count}_{_i_}.png")
                    overlay_img = cv2.imread("image/PCchan.png", cv2.IMREAD_UNCHANGED)

                    x, y = (newCenter[0]+newCenter[2])//2, (newCenter[1]+newCenter[3])//2
                    h, w = overlay_img.shape[:2]
                    lux = x - w//2
                    luy = y - h//2
                    rbx = lux + w 
                    rby = luy + h 

                    # 範囲チェック
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
                        cv2.imwrite(file_path, base_img)


###########################################################

                    _i_ += 1
                    shotFlag = True
                    print("shoted")
                    box_flag = True
                else:
                    count += 1
                    _i_ = 0

        
            #オブジェクトと視線が少し近い
        elif score < 4.5 :
            #3番は短い間隔の音
            current_mode = 3
            b = 255
            start_time = 0
            r = 255
            g = 0

            time_flag = False
            #オブジェクトと視線が離れている
        else :
            #3番は短い間隔の音
            current_mode = 1
            start_time = 0
            g = 255
            r = 255
            b = 0

            time_flag = False


        if shotFlag :
            current_mode = 1

        if box_flag is False:
            cv2.rectangle(new_frame_buffer, (newCenter[0], newCenter[1]), (newCenter[2], newCenter[3]), (b, g, r), 2)
        #cv2.putText(new_frame_buffer, f"ID: {track_id} ,{track_id_to_label[track_id]}",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
       # print(center)

        radius = 30
        bgr_color = (255, 255, 0)
        thickness = 5
        cv2.circle(new_frame_buffer, center, radius, bgr_color, thickness)

        cv2.imshow('Press "q" to exit', new_frame_buffer)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return
        elif key == ord('a'):
            shotFlag = False
            box_flag = False

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

def play_shattersound(filename):
    try:
        sound = pygame.mixer.Sound(filename)
        sound.play()
        while pygame.mixer.get_busy():
            pygame.time.delay(100)
    except pygame.error as e:
        print(f"エラー: {e}")

def tracking(frame , gazePoint):
    global frame_count, prev_detections, prev_tracks, randID , distance_score

    frame_count += 1
    run_detection = (frame_count % frame_skip == 0)

    if run_detection:
        results = model(frame, imgsz=320)  # imgszでさらに高速化

        
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
    #track_id_to_label = {}
    confirmed_tracks = []
    idList = []

    for track in prev_tracks:
        #if not track.is_confirmed():
         #   continue

        #idList.append(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        track_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        # 最も近いクラスを探す
        min_dist = float('inf')
        #matched_class = "unknown"
        #i = 0
        for det in detections :
            #print(f"cls={id1name[det[1]]} clsID={det[1]}")
            if det[1] < len(id1name) and id1name[det[1]] not in ['person'] and id1name[det[1]] not in ['unknown']:#もしそのオブジェクトのクラスがperso , unknown だったらばいちゃ
                dx, dy, dw, dh = det[0]
                dist_x = (dx + dw * 0.5) - track_center[0]
                dist_y = (dy + dh * 0.5) - track_center[1]
                dist = dist_x * dist_x + dist_y * dist_y

                
                setattr(track,"class_id",det[1])
                #det.append(i)
                #i += 1
                #confirmed_tracks.append(det)
                
                if dist < min_dist:
                    min_dist = dist
                    #nearest_det = det
                    #matched_class = id1name[det[1]]
                
        
#ここはよくわからない
     #   if matched_class != "person" and matched_class != "unknown":
      #      track_id_to_label[track_id] = matched_class
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"ID: {track.track_id} ",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #if len(detections) > 5:
       #     for _ in range(len(detections)-5):
          #      del detections[random.randint(0, (len(detections)-1))]



#######---------########


    confirmed_tracks = [track for track in prev_tracks if track.is_confirmed()]

    if randID is None:
        if confirmed_tracks:
            randTrack = random.choice(confirmed_tracks)
            #print(rand_track)
            randID = randTrack.track_id

            distance_score = 0

            print(f"選ばれたID:{randID}")

    idList = [track.track_id for track in confirmed_tracks]
    class_name = None

    if randID in idList:
        for track_2 in confirmed_tracks:
            
            if track_2.track_id == randID:
                class_id = getattr(track,"class_id",None)
                #print(f"抽選済みID:{randID},座標:{track_2.to_ltrb()}")
                x1, y1, x2, y2 = map(int , track_2.to_ltrb())
                coordinaite = [x1, y1, x2, y2]
                class_name = id1name[class_id]
                
    else:
        randID = None
        coordinaite = [0,0,0,0]

#####-----#####-----

    distance_score = 0

    if randID is not None:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # ユークリッド距離を計算
        distance = math.sqrt((gazePoint[0] - center_x) ** 2 + (gazePoint[1] - center_y) ** 2)

        # 任意の正規化やスケーリング（例えば1000で割るなど）
        distance_score = distance / 100.0

        #print(f"distance_score = {distance_score:.2f}")

        print(class_name)
    return frame , coordinaite , distance_score , class_name

async def undistort(cap):
    data = np.load(os.path.join(BASE_DIR, "camera_params.npz"))
    mtx = data["mtx"]
    dist = data["dist"]

    print(mtx,dist,"aaa")

    # 歪み補正を適用
    undistorted = cv2.undistort(cap, mtx, dist, None, mtx)
    return undistorted

import pygame
import time
import threading
import os

# パス設定（どこから実行してもOK）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "beep.wav")

# グローバル制御変数
running = True

loop_channel = None  # mode=4で使うループ用チャンネル

# 初期化
pygame.mixer.init()
beep = pygame.mixer.Sound(sound_path)

def beep_loop():
    
    global loop_channel,current_mode
    print("音声ループ開始" , current_mode)
    

    while running:
        print("音声ループ中" , current_mode)
        if current_mode == 1:
            # 無音モード（停止）
            if loop_channel:
                loop_channel.stop()
                loop_channel = None
            time.sleep(0.1)
            continue

        elif current_mode == 2:
            interval = 0.5
        elif current_mode == 3:
            interval = 0.25
        elif current_mode == 4:
            # 鳴り続ける（ループ再生）
            if not loop_channel or not loop_channel.get_busy():
                loop_channel = beep.play(loops=-1)
            time.sleep(0.1)
            continue
        else:
            interval = 1.0

        # ここから再生処理（モード2 or 3）
        if loop_channel:
            loop_channel.stop()
            loop_channel = None

        channel = beep.play()
        print("音声再生")
        while channel.get_busy():
            time.sleep(0.01)  # 再生終了まで待つ

        time.sleep(interval)

def start_beep_thread():
    print("呼び出すための関数を開始")
    thread = threading.Thread(target=beep_loop)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    start_beep_thread()
    asyncio.run(main())