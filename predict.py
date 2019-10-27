from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys
import numpy as np #npという名前で参照できるように
from PIL import Image

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

def build_model():
        model = Sequential()
        model.add(Conv2D(32,(3,3),padding='same', input_shape=(50,50,3))) #50*50ピクセルのRGBの三色
        model.add(Activation('relu')) #活性化関数で、正のところは通して負のところは0にする
        model.add(Conv2D(32,(3,3))) #フィルタのサイズは3*3
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2))) #poolサイズ　一番大きいサイズを取り出し、より特徴てきな部分を取り出す。
        model.add(Dropout(0.25)) #25%を捨てて、データの偏りをなくす。

        model.add(Conv2D(64,(3,3), padding='same')) #64個のフィルタ（カーネル）を持っている。
        model.add(Activation('relu'))
        model.add(Conv2D(64,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2))) #pooling層の追加
        model.add(Dropout(0.25))

        model.add(Flatten()) #modelのデータを一列に並べる
        model.add(Dense(512)) #全結合処理
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) #半分捨てる
        model.add(Dense(3)) #最後の出力層のノードは3個
        model.add(Activation('softmax')) #それぞれの画像と一致している確率を足しこむと１になるようにする

        opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)  #最適化の手法の宣言 #learning rate   decay 学習レートを減らす

        model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy']) #loss:損失関数 正解と推定値の誤差 これが小さくなるように最適化をする metrics どれぐらい正解したかを保存する

        #モデルのロード
        model = load_model('./animal_cnn.h5')

        return model

def main():
    image = Image.open(sys.argv[1])
    image = image.convert('RGB')
    image = image.resize((image_size,image_size))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model()
    result = model.predict([X])[0]
    predicted = result.argmax() #一番確率が大きいものを取り出す
    percentage = int(result[predicted] * 100)
    print("{0}({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()
