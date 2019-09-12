import mymodule as _

# モデルのロード
model = _.load('model.h5')

# 画像を配列にする
img = _.img_to_array('rice/3.basmati-rice.jpg')

# 画像のリサイズ
img = _.get_resized(img, 32, 32)
print(img.shape)
# 予測
result = model.predict(img.reshape(1, 32, 32, 3)).argmax()

print(result)