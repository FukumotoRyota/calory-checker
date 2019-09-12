import mymodule as _

# ここに学習させたい画像のpathを追加
paths = [
  "./ramen",
  "./rice",
]

# 画像を配列に変換
x, y = _.img_to_npz(paths)

# 画像をリサイズ
# 32のところは適宜変える
x = _.get_resized_many(x, 32, 32)

# 画像の前処理
# ここに色々画像の前処理を書く

# 正規化
x = _.normalize(x)

# One-Hot-Encoding
y = _.oneHotEncoding(y)

# データ分割
x_train, x_test, y_train, y_test = _.split(x, y)

# 学習
# 32のところは上のget_resizedと同じにする
model = _.testLearning(x_train, y_train, 32, 32)

# ファイルに出力
model.save('model.h5', include_optimizer=False)

# 誤答数をチェック
_.error_num(model, x_test, y_test, 32, 32)
