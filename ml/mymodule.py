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

def oneHotEncoding(y):
  return  np_utils.to_categorical(y, num_classes=2).astype('i')

def get_augmented(img):
  if np.random.rand() > 0.5: #50％の確率で画像を左右反転させる
    img = cv2.flip(img, 1)
  if np.random.rand() > 0.5: #50%の確率で画像を左右どちらかに回転させる
    size = (img.shape[0], img.shape[1])
    center = (int(size[0]/2), int(size[1]/2))
    angle = np.random.randint(-30, 31)
    scale=1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.warpAffine(img, rotation_matrix, size)
  return img

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
  model.add(Dense(2))
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

def split(x, y):
  return train_test_split(x, y, train_size=0.75, random_state=10)

def save_npz(x, y):
  np.savez("results.npz", x=x, y=y)

def error_num(model, x_test, y_test, width, height):
  errors = []
  for i in range(len(y_test)):
    pred = model.predict(x_test[i].reshape(1, width, height, 3)).argmax()
    if pred != y_test[i].argmax():
      errors.append(i)
  print(len(y_test), len(errors))