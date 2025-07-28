from PIL import Image

def overlay_images_on_newspaper() -> None:
    
    # 画像を読み込み
    newspaper_img = Image.open("newsPaper.png").convert("RGBA")
    overlay_img1 = Image.open("overlay.png").convert("RGBA")
    overlay_img2 = Image.open("overlay.png").convert("RGBA")
    overlay_img3 = Image.open("overlay.png").convert("RGBA")

    # リサイズサイズを統一（例: 1040x960）
    paste_size = (1040, 960)
    resized_imgs = [
        overlay_img1.resize(paste_size),
        overlay_img2.resize(paste_size),
        overlay_img3.resize(paste_size)
    ]

    # 貼り付ける3か所の座標（左上位置）
    positions = [
        (60, 585),
        (1795, 2100),
        (1795, 3110)
    ]

    # 対応する位置に画像を貼り付け
    for img, pos in zip(resized_imgs, positions):
        newspaper_img.paste(img, pos, img)  # アルファチャンネルを利用

    # 画像を保存
    newspaper_img.save("result.png")



if __name__ == "__main__":
    
    overlay_images_on_newspaper(
        newspaper_path="newsPaper.png",
        overlay_img1_path="overlay.png",
        overlay_img2_path="overlay.png",
        overlay_img3_path="overlay.png",
        output_path="result.png"
    )