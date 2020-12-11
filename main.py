import flask
from flask import Flask, request, render_template
import joblib
from tensorflow.keras.models import Model,load_model
import numpy as np
from scipy import misc
import cv2
import crop
import matplotlib.pyplot as plt
import time

best_model = load_model(filepath = "/Users/harim/Documents/GitHub/brain_tumor_detection/cnn-parameters-improvement-25-0.91.model")

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
  return flask.render_template("index.html")

@app.route('/predict', methods=['POST'])
def make_prediction():
  if request.method == 'POST':

    file = request.files['image']
    if not file:
      return render_template("index.html", label="No Files")

    # reading image
    prediction = []

    IMG_WIDTH, IMG_HEIGHT = (240, 240)

    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    ex_crop_img = crop.crop_brain_contour(img, plot=False)
    ex_crop_img = cv2.resize(ex_crop_img, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    # normalize values
    ex_crop_img = ex_crop_img / 255.

    prediction.append(ex_crop_img)
    prediction = np.array(prediction)

    score = round(float(best_model.predict(prediction)),3)

    result = ''

    if score> 0.4:
      result = "Tumor가 발견되었습니다"
    else:
      result = "Tumor가 발견되지 않았습니다"

    label = "Score:"+str(score)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                    labeltop=False, labelleft=False, labelright=False)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(ex_crop_img)
    plt.tick_params(axis='both', which='both', top=False, bottom=False, left=False, right=False, labelbottom=False,
                    labeltop=False, labelleft=False, labelright=False)
    plt.title('Cropped Image')

    img_name = time.strftime('%H%M%S')+".png"

    plt.savefig("static/"+img_name)
    time.sleep(0.001)
    return render_template("index.html", label=label, img = img_name, res = result)

if __name__=='__main__':
  app.run(host='0.0.0.0', port=8888, debug=True)
