from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np #npという名前で参照できるように

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

#メインの関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy",allow_pickle=True)
    X_train = X_train.astype("float") / 256 #256階調の整数値は0か1に入っている方がニューラルネットワークではぶれが少ない
    X_test = X_test.astype("float") / 256 #floatとして読み込む
    y_train = np_utils.to_categorical(y_train, num_classes) #one-hot-vector 正解は1 他はぜろ
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train) #trainを実行
    model_eval(model, X_test, y_test) #評価をする


# See [https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py]
def model_train(X, y):
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same', input_shape=X.shape[1:])) #X_train.shape -> (450, 50, 50, 3) 50*50*3のデータが450個ある。個数は不要なので、2番目以降を取得
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

    model.fit(X, y, batch_size=32, epochs=100) # bat_size:一回のトレーニング（エポック）に使うデータの個数  number of epoch:トレーニングを何セットするのか  epoch数を増やすと、制度は上がる。

    #モデルの保存
    model.save('./animal_cnn_aug.hs')

    return model

def model_eval(model, X, y):   #Xのテストとyのテストが入る
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy', scores[1])

if __name__ == "__main__": #ファイルを他のプログラムから参照するために、このプログラムが直接呼ばれたときのみmainを実行する そうでなければ各関数を引用できるようにする
    main()
