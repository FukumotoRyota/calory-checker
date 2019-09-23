import numpy as np
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn import metrics
from keras import regularizers
from keras.layers.convolutional import ZeroPadding2D, MaxPooling2D
from keras.layers import BatchNormalization, Dropout, AveragePooling2D, merge
from keras.callbacks import EarlyStopping
import matplotlib as plt


def load(path):
  return load_model(path, compile=False)

def img_to_array(path):
  img_array = cv2.imread(path)
  return img_array

# pathsに['./ramen', './rice', ...]のような形でpathを渡す
def img_to_npz(paths):
  results = []
  labels = []
  for index, path in enumerate(paths):
    images = glob("{}/*".format(path))
    print(len(images))
    for img in images:
      img_array = img_to_array(img)
      results.append(img_array)
      labels.append(index)
  x = np.array(results)
  y = np.array(labels)
  return x, y

def normalize(x):
  x = x.astype('f')
  x /= 255
  return x

def oneHotEncoding(y, num):
  return  np_utils.to_categorical(y, num_classes=num).astype('i')

# def get_augmented(img):
#   if np.random.rand() > 0.5: #50％の確率で画像を左右反転させる
#     img = cv2.flip(img, 1)
#   if np.random.rand() > 0.5: #50%の確率で画像を左右どちらかに回転させる
#     size = (img.shape[0], img.shape[1])
#     center = (int(size[0]/2), int(size[1]/2))
#     angle = np.random.randint(-30, 31)
#     scale=1.0
#     rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
#     img = cv2.warpAffine(img, rotation_matrix, size)
#   return img

def get_resized(img, width, height):
  resized = cv2.resize(img, (width, height)) #画像を(width, height)にリサイズする
  return resized

def get_resized_many(images, width, height):
  result = []
  for img in images:
    resized = get_resized(img, width, height)
    result.append(resized)
  result = np.array(result)
  return result

# def process_image(images):
#   result=list()
#   for image in images

#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) #ヒストグラム平坦化
#     cl = clahe.apply((image * 255).astype('uint8'))
#     cl = cl.astype(np.float32) / 255

#     blurred = cv2.GaussianBlur(cl, (3, 3), 0) #平滑化

#     resized = get_resized(blurred) #リサイズ

#     augmented = get_augmented(resized) #水増し

#     augmented = augmented[:, :, np.newaxis]

#     result.append(augmented.tolist())

#     return np.array(result)

def testLearning(x, y, width, height):

  model = Sequential()

  model.add(Conv2D(filters=64, input_shape=(width, height, 3), kernel_size=(4, 4), strides=(1, 1), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Activation('relu'))

  model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='same'))
  model.add(MaxPool2D(pool_size=(2, 2)))
  model.add(Activation('relu'))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dense(len(y[0])))
  model.add(Activation('softmax'))

  # 同様に学習前にコンパイルします。
  model.compile(loss='categorical_crossentropy',
                optimizer=SGD(0.01),  # 学習率を0.01に指定
                metrics=['accuracy'])

  # ミニバッチに含まれるサンプル数を指定
  batch_size = 120

  # epoch数を指定
  n_epoch = 30

  # 学習を開始します。
  hist = model.fit(x,
                  y,
                  epochs=n_epoch,
                  validation_data=(x, y),
                  verbose=1,
                  batch_size=batch_size)

  # 返り値
  return model

def CNNmodel(X_train, y_train, X_test, y_test, width, height, kernel, batch, epoch):
    model = Sequential()

    model.add(Conv2D(filters=64, input_shape=(width, height, 3), kernel_size=(kernel, kernel), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=(kernel, kernel), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(filters=128, kernel_size=(kernel, kernel), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(len(y_train[0])))
    model.add(Activation('softmax'))

    model.summary()

    # 同様に学習前にコンパイルします。
    model.compile(loss='categorical_crossentropy',
                  optimizer="Adam",  # 学習率を0.01に指定
                  metrics=['accuracy'])

    # ミニバッチに含まれるサンプル数を指定
    batch_size = batch

    # epoch数を指定
    n_epoch = epoch

    # 学習を開始します。
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

    # 学習の際にEarlyStoppingを適用
    hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch,
                     validation_data=(X_test, y_test), verbose=1,
                     callbacks=[es]) # EarlyStoppingを適用

    

    # 性能指標を確認
    # temp_y = []
    # for y in y_test:
    #   temp_y.append(np.argmax(y))
    # temp_y = np.array(temp_y)
    # print('accuracy: %.3f' % metrics.accuracy_score(temp_y, model.predict(X_test).argmax(axis=1)))
    # print('recall: %.3f' % metrics.recall_score(temp_y, model.predict(X_test).argmax(axis=1), average='macro'))
    # print('precision: %.3f' % metrics.precision_score(temp_y, model.predict(X_test).argmax(axis=1), average='macro'))
    # print('f1_score: %.3f' % metrics.f1_score(temp_y, model.predict(X_test).argmax(axis=1), average='macro'))

    # 返り値
    return model

def AlexNet(ROWS=32, COLS=32):
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(ROWS, COLS, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３−５畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def inception_module(x, params, concat_axis,
                     subsample=(1, 1), activation='relu',
                     border_mode='same', weight_decay=None):

    (branch1, branch2, branch3, branch4) = params

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    pathway1 = Conv2D(branch1[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False)(x)


    pathway2 = Conv2D(branch2[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False)(x)
    pathway2 = Conv2D(branch2[1], 3, 3,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False)(pathway2)


    pathway3 = Conv2D(branch3[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False)(x)
    pathway3 = Conv2D(branch3[1], 5, 5,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False)(pathway3)


    pathway4 = MaxPooling2D(pool_size=(1, 1))(x)
    pathway4 = Conv2D(branch4[0], 1, 1,
                             subsample=subsample,
                             activation=activation,
                             border_mode=border_mode,
                             W_regularizer=W_regularizer,
                             b_regularizer=b_regularizer,
                             bias=False,
                             )(pathway4)

    return merge([pathway1, pathway2, pathway3, pathway4],
                 mode='concat', concat_axis=concat_axis)

#Model
def GoogleNet(ROWS=277, COLS=277):
    input = keras.layers.Input(shape=(ROWS, COLS, 3))
    x = Conv2D(64, input_shape=(ROWS, COLS, 3),
                 kernel_size=(7, 7),
                 strides=(1, 1))(input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(3, 3))(x)
    x = Conv2D(64, kernel_size=(1, 1))(x)
    x = Conv2D(192, kernel_size=(3, 3), strides=(1, 1))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(3, 3))(x)

    CONCAT_AXIS = 3
    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                     concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                     concat_axis=CONCAT_AXIS)

    x = MaxPool2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = ZeroPadding2D(padding=(1, 1))(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                     concat_axis=CONCAT_AXIS)


    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                     concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                     concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
                     concat_axis=CONCAT_AXIS)


    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                     concat_axis=CONCAT_AXIS)
    x = MaxPool2D(pool_size=(1, 1), strides=(1, 1))(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                     concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                     concat_axis=CONCAT_AXIS)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000)(x)
    x = Activation('linear')(x)
    x = Dense(1000)(x)
    x = Activation('softmax')(x)

    model = keras.models.Model(inputs=input, outputs=x)
    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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

def split(x, y):
  return train_test_split(x, y, train_size=0.75, random_state=10)

def save_npz(x_train, x_test, y_train, y_test):
  np.savez("results.npz", x_train, x_test, y_train, y_test)

def error_num(model, x_test, y_test, width, height):
  errors = []
  for i in range(len(y_test)):
    pred = model.predict(x_test[i].reshape(1, width, height, 3)).argmax()
    if pred != y_test[i].argmax():
      errors.append(i)
  print(len(y_test), len(errors))

# data augmentationを行う関数
def get_augmented(img, random_crop=4):
  # 左右反転のノイズを加える
  if np.random.rand() > 0.5:
    img = np.fliplr(img)
  # 左右どちらかに30度回転させる
  if np.random.rand() > 0.5:
    size = (img.shape[0], img.shape[1])
    # 画像の中心位置(x, y)
    center = (int(size[0]/2), int(size[1]/2))
    # 回転させたい角度
    angle = np.random.randint(-30, 30)
    # 拡大比率
    scale = 1.0
    # 回転変換行列の算出
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # 並進移動
    img = cv2.warpAffine(img, rotation_matrix, size, cv2.INTER_CUBIC)
  return img

# 画像の前処理を行う関数
def process_image(image):
  # image = image.astype('f')
  # サイズをVGG16指定のものに変換する
  image = cv2.resize(image, (224, 224))
  # RGBからそれぞれvgg指定の値を引く(mean-subtractionに相当)
  # image[:, :, 0] -= 100
  # image[:, :, 1] -= 116.779
  # image[:, :, 2] -= 123.68
  # 0-1正規化
  # image /= image.max()
  # augmentation
  # image = get_augmented(image)
  return image
