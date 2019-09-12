from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array, load_img
import base64
from mysqlcon import MySQLCon
import cv2

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
  if request.method == "GET":
    return render_template("index.html", message="Hello World!")
  else:
    img = request.files['picture']
    img_b64 = base64.b64encode(img.read()).decode("utf-8")
    img_array = cv2.resize(img_to_array(load_img(img)), (64, 64))
    print(img_array.shape)
    # 予測

    # 照合
    connection = MySQLCon(host='localhost', port=3306, user='root', passwd='password', db='cal')
    connection.query("SELECT * FROM foods WHERE id=1;")
    return render_template("index.html", message="Image received!", img=img_b64)

