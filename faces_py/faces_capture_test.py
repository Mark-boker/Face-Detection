import cv2
import os
import numpy as np

# 加载训练好的人脸识别模型
model = cv2.face.LBPHFaceRecognizer_create()
model.read("mafance_model.yml")

# 加载人脸检测分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

# 打开摄像头
capture = cv2.VideoCapture(0)

# 定义标签映射字典
labels_test = {"0": "mark"}

while True:
    # 读取摄像头捕获的图像
    retval, image = capture.read()

    if not retval:
        break

    # 将图像转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用人脸检测分类器检测图像中的人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 遍历检测到的人脸
    for (x, y, w, h) in faces:
        # 在原图像上绘制人脸矩形框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 使用人脸识别模型预测人脸的标签和置信度
        label, confidence = model.predict(gray[y:y + h, x:x + w])

        # 将标签转换为对应的姓名
        name = labels_test.get(str(label), "Unknown")

        # cv2.putText(image, f"{name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 在图像上显示识别结果的标签和置信度
        if confidence < 50:
            cv2.putText(image, f"{name} ({confidence:.2f}) - success", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
        else:
            cv2.putText(image, f"{name} ({confidence:.2f})- failed", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
    # 显示图像
    cv2.imshow("Face Recognition", image)

    key = cv2.waitKey(1) & 0xFF

    # 如果按下了 Esc 键（ASCII 码为 27）或 'q' 键，则退出循环
    if key == 27 or key == ord('q'):
        break

# 释放摄像头资源
capture.release()
# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
