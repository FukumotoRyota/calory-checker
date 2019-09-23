from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import base64
from mysqlcon import MySQLCon
import cv2
import tensorflow as tf
import numpy as np

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
  return image

app = Flask(__name__)
model = load_model('../model/model_90.h5', compile=False)
graph = tf.get_default_graph()

@app.route("/", methods=["GET", "POST"])
def index():
  if request.method == "GET":
    return render_template("index.html", message="画像を送信してください")
  else:
    img = request.files['picture']
    img_b64 = base64.b64encode(img.read()).decode("utf-8")
    img_array = img_to_array(load_img(img))
    # 画像の加工
    print(img_array.shape)
    img_array = process_image(img_array)
    print(img_array.shape)
    # 予測
    global graph
    with graph.as_default():
      pred = model.predict(img_array.reshape(1, 224, 224, 3)).argmax()
    # 照合
    # connection = MySQLCon(host='localhost', port=3306, user='root', passwd='password', db='cal')
    # connection.query("SELECT * FROM foods WHERE id=1;")
    if pred == 0:
      message = "シュウマイです。100gあたり215.1kcalです。"
    elif pred == 1:
      message = "チャーハンです。一人分で629kcalです。"
    elif pred == 2:
      message = "餃子です。100gあたり196.9kcalです。"
    elif pred == 3:
      message = "ラーメンです。一人分で443kcalです。"
    elif pred == 4:
      message = "エビチリです。100gで128kcalです。"
    return render_template("index.html", message=message, img=img_b64)

app.run(host='0.0.0.0', port=3000, threaded=True)
