import cv2
import os
import numpy as np

# 打开摄像头
capture = cv2.VideoCapture(0)

# 判断摄像头是否打开
if capture.isOpened():
    print("摄像头正常")

    count = 0
    while count < 5:
        retval, image = capture.read()

        if retval:
            count += 1
            print("当前抓到第", count, "张图像")
            # 检测分类器
            face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

            faces = face_cascade.detectMultiScale(image)
            num = 0
            for (x, y, w, h) in faces:
                img_copy = image[y: y + h, x: x + w]
                img_resize = cv2.resize(img_copy, dsize=(200, 200))

                num += 1
                #用于创建目录,如果目录已经存在，exist_ok=True 参数会防止函数抛出异常。
                os.makedirs("faces", exist_ok=True)
                cv2.imwrite("faces\\myface_%02d_%02d.jpg"%(count ,num), img_resize)

                cv2.rectangle(image,
                              (x, y),
                              (x + w, y + h),
                              color=(255, 0, 0),
                              thickness=2)
            cv2.imshow("capture image", image)
            cv2.waitKey(2000)

        else:
            continue

    capture.release()

else:
    print("摄像头未打开")