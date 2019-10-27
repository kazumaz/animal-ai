# ニューラルネットワークの定義用のモジュール
from keras.models import Sequential
# 畳み込みやプーリングの処理用モジュール
from keras.layers import Conv2D, MaxPooling2D
# 活性化関数、ドロップアウト処理、データを一次元に返還する、全結合層のていぎ
from keras.layers import Activation, Dropout, Flatten, Dense
# numpyデータを扱うモジュール
from keras.utils import np_utils
import keras #TensorFlowやTheano上で動くニューラルネットワークライブラリ
import numpy as np #npという名前で参照できるように

classes = ["monkey","boar","crow"]
num_classes = len(classes)
image_size = 50

#メインの関数を定義する
def main():
    #生成したnumpy配列の読み込みを行う
    X_train, X_test, y_train, y_test = np.load("./animal.npy",allow_pickle=True)
    #データの正規化(0か1に収める)
    #整数を浮動小数点に変更して、256でわる。
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    # one-hot-vectorに変換する
    # 正解は1,それ以外は0にする
    # [0,1,2]を[1.0.0][0.1.0][0.0.1]に変換する
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train) #trainを実行
    model_eval(model, X_test, y_test) #評価をする


# See [https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py]
def model_train(X, y):
    # モデルの作成
    model = Sequential()
    # 第一層の定義
    # 32個の3*3のフィルター
    # 畳み込み結果が同じサイズになるように畳み込む
    # 入力の形を50*50*3(0番目は枚数なので不要)
    # X.shape -> (450, 50, 50, 3) 50*50*3のデータが450個ある。個数は不要なので、2番目以降を取得
    model.add(Conv2D(32,(3,3),padding='same', input_shape=X.shape[1:]))
    model.add(Activation('relu')) #活性化関数で、正のところは通して負のところは0にする
    # 2層目
    model.add(Conv2D(32,(3,3))) #フィルタのサイズは3*3
    model.add(Activation('relu'))
    #poolサイズ　一番大きいサイズを取り出し、より特徴てきな部分を取り出す。
    model.add(MaxPooling2D(pool_size=(2,2)))
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
    model.add(Dense(3)) #最後の出力層のノードは3個(画像が3つなので)
    model.add(Activation('softmax')) #それぞれの画像と一致している確率を足しこむと１になるようにする

    #最適化の手法の宣言 #learning rate   decay 学習レートを減らす
    opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)
    #loss:損失関数 正解と推定値の誤差 これが小さくなるように最適化をする metrics どれぐらい正解したかを保存する
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    # bat_size:一回のトレーニング（エポック）に使うデータの個数  number of epoch:トレーニングを何セットするのか
    model.fit(X, y, batch_size=32, epochs=100)

    #モデルの保存
    model.save('./animal_cnn.hs')

    return model

def model_eval(model, X, y):   #Xのテストとyのテストが入る
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy', scores[1])

# ファイルを他のプログラムから参照するために、このプログラムが直接呼ばれたときのみmainを実行する
# そうでなければ各関数を引用できるようにする
if __name__ == "__main__":
    main()
