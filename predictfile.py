from keras.models import Sequential, load_model
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename #file名に危険な名前がある場合はサニタイズするためのもの

import keras, sys
import numpy as np #npという名前で参照できるように
from PIL import Image

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENTIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'} #受け付ける拡張子

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    #共にチェックが該当すれば1を返却する いずれかがNGなら0を返す
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/', methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model = load_model('./animal_cnn.h5')

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)

            return "ラベル： " + classes[predicted] + ", 確率："+ str(percentage) + " %"

            # return redirect(url_for('uploaded_file', filename=filename)) #redirectして画像を表示する
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title></head>
    <body>
    <h1>ファイルをアップロードして判定しよう!</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
