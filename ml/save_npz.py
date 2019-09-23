import mymodule as _
import numpy as np
import cv2

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
  image = image.astype('f')
  # サイズをVGG16指定のものに変換する
  image = cv2.resize(image, (224, 224))
  # RGBからそれぞれvgg指定の値を引く(mean-subtractionに相当)
  image[:, :, 0] -= 100
  image[:, :, 1] -= 116.779
  image[:, :, 2] -= 123.68
  # 0-1正規化
  image /= image.max()
  # augmentation
  image = get_augmented(image)
  return image

# ここに学習させたい画像のpathを追加
paths = [
  "./editchnfood/dumpl",
  "./editchnfood/frrice",
  "./editchnfood/gyoza",
  "./editchnfood/ramen",
  "./editchnfood/shrimp",
  "./editchnfood/ビビンバ",
  "./editchnfood/回鍋肉",
  "./editchnfood/春巻き",
  "./editchnfood/酢豚",
  "./editchnfood/麻婆豆腐",
]

# 画像を配列に変換
x, y = _.img_to_npz(paths)

# Noneを削除
temp_x = []
temp_y = []
for img, label in zip(x, y):
  if img is None:
    continue
  temp_x.append(img)
  temp_y.append(label)
x = np.array(temp_x)
y = np.array(temp_y)

# データ分割
x_train, x_test, y_train, y_test = _.split(x, y)

# X_trainにaugmetation処理
X_train_list = []
for img in x_train:
  X_train_list.append(process_image(img))
X_train_aug = np.array(X_train_list) # 扱いやすいようlistをndarrayに変換

# X_testにaugmetation処理
X_test_list = []
for img in x_test:
  X_test_list.append(process_image(img))
X_test_aug = np.array(X_test_list) # 扱いやすいようlistをndarrayに変換

# # One-Hot-Encoding
y_train = _.oneHotEncoding(y_train, len(paths))
y_test = _.oneHotEncoding(y_test, len(paths))

# save as npz
print(X_train_aug.shape, X_test_aug.shape, y_train.shape, y_test.shape)
_.save_npz(X_train_aug, X_test_aug, y_train, y_test)