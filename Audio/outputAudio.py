import pygame
import time
import threading
import os

# パス設定（どこから実行してもOK）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "beep.wav")

# グローバル制御変数
current_mode = 3
running = True
loop_channel = None  # mode=4で使うループ用チャンネル

# 初期化
pygame.mixer.init()
beep = pygame.mixer.Sound(sound_path)

def beep_loop():
    global loop_channel

    while running:
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
        while channel.get_busy():
            time.sleep(0.01)  # 再生終了まで待つ

        time.sleep(interval)

def start_beep_thread():
    thread = threading.Thread(target=beep_loop)
    thread.daemon = True
    thread.start()

# 実行例
if __name__ == "__main__":
    print("再生スレッドを開始します（Ctrl+Cで終了）")
    start_beep_thread()

    try:
        while True:
            print(f"\n現在のモード：{current_mode}")
            mode = input("モードを選んでください (1=無音, 2=長め, 3=短め, 4=連続): ")
            if mode in ["1", "2", "3", "4"]:
                current_mode = int(mode)
            else:
                print("無効な入力です。")
    except KeyboardInterrupt:
        print("\n終了します")
        running = False
        if loop_channel:
            loop_channel.stop()
        time.sleep(1)
