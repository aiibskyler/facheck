import face_recognition
import cv2
import os


def face_validate(base_img, image):
    # 存储知道的特征值
    known_encodings = []
    load_image = face_recognition.load_image_file(base_img)  # 加载图片
    image_face_encoding = face_recognition.face_encodings(load_image)[0]  # 获得128维特征值
    known_encodings.append(image_face_encoding)
    # print(known_encodings)

    # while True:
    face_locations = face_recognition.face_locations(image)  # 获得所有人脸位置
    face_encodings = face_recognition.face_encodings(image, face_locations)  # 获得人脸特征值
    face_names = []  # 存储出现在画面中人脸的名字
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        if True in matches:
            return True
        else:
            return False

        # 将捕捉到的人脸显示出来
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)  # 画人脸矩形框
    #     # 加上人名标签
    #     cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # cv2.imshow('frame', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
