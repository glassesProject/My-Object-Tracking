# image_generator.py
import json
import random
import os
import time
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAIT_FLAG_PATH = os.path.join(BASE_DIR, 'state', 'wait_flag.txt')
STATUS_FILE_PATH = os.path.join(BASE_DIR, 'state', 'external_status.json')

def update_loop():
    while True:
        
        with open(STATUS_FILE_PATH, 'r') as f:
            config = json.load(f)
        
        #生成フラグがfalseなら
        if not config.get("iscreating"):
            print("画像生成停止中：ボタンが押されるのを待っています")
                    # ランダムにTrue or Falseを生成
            status_value = random.choice([True, False])

            # 辞書にして保存
            status_data = {"status": status_value}

            # JSONファイルに書き込む
            with open(STATUS_FILE_PATH, 'w') as f:
                json.dump(status_data, f)

            print(f"status: {status_value}")
        
            time.sleep(1)
            continue

        # 画像生成
        img = Image.new('RGB', (400, 300), color='white')
        current_time = time.strftime("%H:%M:%S")
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Now: {current_time}", fill='black')
        img.save(os.path.join(BASE_DIR, 'static/images/generated_image.jpg'))
        print("画像更新:", current_time)

        # ✅ フラグは作らない！再びループで続行（Flask側で制御）
        time.sleep(10)

if __name__ == '__main__':
    update_loop()
