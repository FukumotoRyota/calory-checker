# 必要なライブラリの読み込み
import pickle
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import keras
from keras.models import Sequential           #層構造のモデルを定義するためのメソッド
from keras.layers import Dense, Activation          #Denseは層の生成メソッド、Activationは活性化関数を定義するためのメソッド
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Input, Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn import metrics

# 精度と誤差をグラフ描画
def plot_history_loss(hist):
    # 損失値(Loss)の遷移のプロット
    plt.plot(hist.history["loss"],label="loss for training")
    plt.plot(hist.history["val_loss"],label="loss for validation")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.show()

def plot_history_acc(hist):
    # 精度(Accuracy)の遷移のプロット
    plt.plot(hist.history["acc"],label="accuracy for training")
    plt.plot(hist.history["val_acc"],label="accuracy for validation")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.show()

def learning(X_train, X_test, Y_train, Y_test):
  # 入力画像のサイズを指定
  input_tensor = Input(shape=(32, 32, 3))

  # 学習済みモデルの読み込み
  # ダウンロードに数十分かかります
  base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

  # 必要なパラメータの追加
  input_height = 32
  input_width = 32
  n_class = 5

  # 学習済みモデルに加える全結合層部分を定義
  # 最終層はノード数がクラスラベルの数に一致していないのでスライシングで取り除く
  top_model = Sequential()
  top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
  top_model.add(Dense(256))
  top_model.add(Activation('relu'))
  top_model.add(Dropout(0.5))
  top_model.add(Dense(n_class))
  top_model.add(Activation('softmax'))

  # base_modelとtop_modelを接続
  model = Model(input=base_model.input, output=top_model(base_model.output))

  # 畳み込み層の重みを固定（学習させない）
  for layer in model.layers[:15]:
          layer.trainable = False

  # モデルのコンパイル
  model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=0.0001),
                metrics=['accuracy'])

  batch_size = 100
  n_epoch = 100 # 簡単に動作確認をするため､epochを1に設定

  # 同じように学習完了後、histから結果を確認できます。
  hist = model.fit(X_train,
                  Y_train,
                  epochs=n_epoch,
                  validation_data=(X_test, Y_test),
                  verbose=1,
                  batch_size=batch_size)

  # 性能指標を確認
  temp_y = []
  for y in y_test:
    temp_y.append(np.argmax(y))
  temp_y = np.array(temp_y)
  print('accuracy: %.3f' % metrics.accuracy_score(temp_y, model.predict(x_test).argmax(axis=1)))
  print('recall: %.3f' % metrics.recall_score(temp_y, model.predict(x_test).argmax(axis=1), average='macro'))
  print('precision: %.3f' % metrics.precision_score(temp_y, model.predict(x_test).argmax(axis=1), average='macro'))
  print('f1_score: %.3f' % metrics.f1_score(temp_y, model.predict(x_test).argmax(axis=1), average='macro'))

  # history
  plot_history_loss(hist)
  plot_history_acc(hist)

  return model