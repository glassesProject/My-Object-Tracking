from PIL import Image

# --- 設定項目 ---

# 1. 画像ファイルのパスを指定
base_image_path = r"C:\Users\sk062\Downloads\newsPaper.png"  # 背景の新聞テンプレート
insert_image1_path = r"C:\Users\sk062\Downloads\no0_0.png" # image1に貼り付ける画像
output_path = r'C:\Users\sk062\Downloads\completed_newspaper.png'       # 完成した画像の保存先

# 2. 各画像の貼り付け位置 (左上のX, Y座標) とサイズ (幅, 高さ) を指定
#    ※これらの値はテンプレートに合わせて調整してください
pos_image1 = (40, 140)  # image1の貼り付け先座標 (X, Y)
size_image1 = (400, 255) # image1のサイズ (幅, 高さ)

pos_image2 = (465, 480) # image2の貼り付け先座標 (X, Y)
size_image2 = (390, 255) # image2のサイズ (幅, 高さ)

pos_image3 = (465, 755) # image3の貼り付け先座標 (X, Y)
size_image3 = (390, 255) # image3のサイズ (幅, 高さ)

# --- 画像処理の実行 ---

try:
    # 背景画像と貼り付ける画像を開く
    base_image = Image.open(base_image_path).convert("RGBA")
    insert_image1 = Image.open(insert_image1_path).convert("RGBA")
    insert_image2 = Image.open(insert_image1_path).convert("RGBA")
    insert_image3 = Image.open(insert_image1_path).convert("RGBA")

    # 貼り付ける画像を指定されたサイズにリサイズ
    insert_image1 = insert_image1.resize(size_image1)
    insert_image2 = insert_image2.resize(size_image2)
    insert_image3 = insert_image3.resize(size_image3)

    # 背景画像に各画像を貼り付け
    base_image.paste(insert_image1, pos_image1, insert_image1)
    base_image.paste(insert_image2, pos_image2, insert_image2)
    base_image.paste(insert_image3, pos_image3, insert_image3)

    # 完成した画像を保存
    base_image.save(output_path)
    print(f"画像が '{output_path}' として正常に保存されました。")

except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。パスを確認してください。 - {e}")
except Exception as e:
    print(f"エラーが発生しました: {e}")