import cv2
import io
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from service.core_validate import face_validate

app = Flask(__name__)


@app.route('/face/check', methods=['POST'])
def face_check():
    base_img = request.files['base_file']
    uncheck_file = request.files['uncheck_file']
    img = uncheck_file.read()
    # im1 = Image.open(img)
    img_cv2 = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    if face_validate(base_img, img_cv2):
        return "验证成功"
    else:
        return "验证失败"


if __name__ == '__main__':
    app.run()  # 执行flask的运行
