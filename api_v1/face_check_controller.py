import cv2
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from service.core_validate import face_validate

app = Flask(__name__)


@app.route('/face/check', methods=['POST'])
def face_check():
    f = request.files['file']
    img = f.read()
    # im1 = Image.open(img)
    img_cv2 = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    face_validate("../static/images/", img_cv2)
    # while True:
    #     cv2.imshow('img', img_cv2)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         return
    # im = Image.fromarray(img_cv2)
    # image = cv2.imread(im)
    # byte_stream = io.BytesIO(img)
    # ret, img = Image.open(byte_stream)
    return 'Send Img Test'


if __name__ == '__main__':
    app.run()  # 执行flask的运行
