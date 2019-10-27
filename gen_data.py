from PIL import Image #ピロー　画像処理ライブラリ
#os OS操作が可能になる、ファイル一覧取得など。　
#globは引数に指定されたパターンにマッチするファイルパス名を取得できる。
import os, glob
import numpy as np #次元配列を扱う数値演算ライブラリ
from sklearn import datasets, svm, metrics #サイキットラーン：機械学習用ライブラリ
from sklearn.model_selection import train_test_split
from sklearn import model_selection

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50 #50ピクセルにするため

#画像の読み込みを実施する
X = []
Y = []

################################
# enumerateの使い方
# for i, name in enumerate(l):
#     print(i, name)
#
#  0 Alice
#  1 Bob
#  2 Charlie
#
################################

for index, classlabel in enumerate(classes): #1のmonkey, 2のboarというようにお取り出す
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg") #パターン一致している画像を持ってくる
    for i, file in enumerate(files):
        if i >= 300: break
        image = Image.open(file)
        image = image.convert("RGB") #イメージをRGBに変換
        image = image.resize((image_size, image_size)) #サイズを50ピクセルへ変更
        data = np.asarray(image) #数字の配列にしちゃう
        X.append(data)
        Y.append(index) #0,1,2が入っていく

# np.arrayとnp.asarrayの違いは [https://punhundon-lifeshift.com/array_asarray]

X = np.array(X)  #TensorFlowが扱いやすいデータ側（numpyのarray）に変更する
Y = np.array(Y)

# 機械学習させるためのデータ「train」と、その後のテストデータに分割する。
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./animal.npy", xy) #npの配列をテキストファイルとして保存する
