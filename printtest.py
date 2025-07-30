import win32print
import win32ui
import win32con
from PIL import Image, ImageWin, ImageEnhance

def print_png(filename: str, printer_name: str = None):
    """
    PNG画像を白黒感熱プリンタで正しく印刷する（A4縦向き）。
    :param filename: 印刷対象のPNGファイルパス
    :param printer_name: プリンタ名（未指定なら既定を使用）
    """

    # 既定プリンタを使用（未指定の場合）
    if printer_name is None:
        printer_name = win32print.GetDefaultPrinter()

    # 画像読み込み
    img = Image.open(filename)

    # 白黒プリンタ向け：グレースケール → コントラスト強調 → 2値化
    img = img.convert("L")  # グレースケール
    img = ImageEnhance.Contrast(img).enhance(2.0)  # コントラスト2倍
    img = img.point(lambda x: 0 if x < 128 else 255, mode='1')  # 白黒2値化

    # プリンタデバイス作成
    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(printer_name)
    hDC.SetMapMode(win32con.MM_TEXT)  # ピクセル単位で描画

    # 印刷可能サイズ（ピクセル）取得
    printable_width = hDC.GetDeviceCaps(win32con.HORZRES)
    printable_height = hDC.GetDeviceCaps(win32con.VERTRES)

    # 横幅に合わせて画像サイズを縮小（高さ比率維持）
    target_width = printable_width
    target_height = int(img.height * target_width / img.width)
    img = img.resize((target_width, target_height))

    # 印刷処理
    dib = ImageWin.Dib(img)
    hDC.StartDoc(filename)
    hDC.StartPage()

    # 左上から描画（マージン未調整でよければ (0,0) でOK）
    y_offset = -100  # 負の値で上方向に移動
    dib.draw(hDC.GetHandleOutput(), (-80, -100, target_width, target_height - 150))

    hDC.EndPage()
    hDC.EndDoc()
    hDC.DeleteDC()


# -------------------------------
# テスト実行用コード（ここから）
# -------------------------------
if __name__ == "__main__":
    import os

    # テスト対象のファイル
    test_file = "newsPaper.png"  # ← ここに印刷したいファイル名を指定

    if not os.path.exists(test_file):
        print(f"ファイルが見つかりません: {test_file}")
    else:
        print("印刷を開始します...")
        try:
            print_png(test_file)
            print("印刷が完了しました。")
        except Exception as e:
            print(f"印刷中にエラーが発生しました: {e}")
