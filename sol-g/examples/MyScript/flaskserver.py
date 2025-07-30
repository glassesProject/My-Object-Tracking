from flask import Flask, jsonify, render_template
import os
import json

app = Flask(__name__)

#最初に呼ばれる関数
@app.route('/')
def index():

    return render_template('index.html')

#Javascriptから叩くと画像ファイルの更新日時を返す
@app.route('/image_status')
def image_status():
    print("画像の更新時間をreturn")
    IMAGE_PATH = 'static/images/generated_image.png'
    if os.path.exists(IMAGE_PATH):
        timestamp = os.path.getmtime(IMAGE_PATH)
        return jsonify({'last_updated': timestamp})
    return jsonify({'error': 'Image not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True) 
    #デバッグモードにしておく
    #app.run(debug=True)