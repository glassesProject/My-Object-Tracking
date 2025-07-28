from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

# Chromeドライバーの設定
chrome_options = Options()
chrome_options.add_argument("--headless")  # ヘッドレスモード（GUIなし）
service = Service('path/to/chromedriver')  # ChromeDriverのパスを指定

# ブラウザを開く
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # 指定のURLにアクセス
    url = "https://example.com"  # 目的のURLを指定
    driver.get(url)
    
    # タイムディレイを30秒挟む
    time.sleep(30)

    # クラス名が"overview_table"のテーブルを取得
    table = driver.find_element(By.CLASS_NAME, "overview_table")
    
    # テーブルの内容を取得（例: テキストを出力）
    print(table.text)

finally:
    # ブラウザを閉じる
    driver.quit()
