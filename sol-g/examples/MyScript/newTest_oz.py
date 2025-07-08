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

from utils.server_info import get_ip_and_port
from ganzin.sol_sdk.streaming.gaze_stream import GazeData
from ganzin.sol_sdk.synchronous.models import StreamingMode
from ganzin.sol_sdk.synchronous.sync_client import SyncClient
from ganzin.sol_sdk.utils import find_nearest_timestamp_match



#YOLOの読み込み
model = YOLO("yolov8n.pt").to('cuda')

# DeepSORT初期化
tracker = DeepSort(max_age=30)

print(type(recv_video))
#cv2.imshow("testTracking" , recv_video)