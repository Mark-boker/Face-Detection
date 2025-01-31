import cv2
import os
import numpy as np

# 初始化两个列表，用于存储人脸图像数据和对应的标签
faces = []
labels = []

# 定义一个字典，用于存储标签的映射关系
labels_test = {"0": "meixi"}

# 遍历"meixi"目录下的所有文件
for f in os.listdir("meixi"):
    # 如果文件不是jpg格式的图像文件，则跳过
    if '.jpeg' not in f:
        continue

    # 使用cv2.imread()函数读取图像文件，并将其转换为灰度图像
    image = cv2.imread("meixi\\" + f, cv2.IMREAD_GRAYSCALE)

    # 将读取到的灰度人脸图像添加到faces列表中
    faces.append(image)

    # 将标签0添加到labels列表中，表示这些人脸图像都属于标签0对应的类别
    labels.append(0)

# 如果faces列表中至少有一个元素（即至少读取到一张人脸图像）
if len(faces) > 0:
    # 创建一个LBPHFaceRecognizer对象，用于人脸识别
    recognizer = cv2.face.LBPHFaceRecognizer.create()

    # 使用recognizer.train()方法训练人脸识别模型
    recognizer.train(faces, np.array(labels))

    # 使用recognizer.save()方法将训练好的模型保存到"meixi_face_model.yml"文件中
    model_path = "meixi_face_model.yml"
    recognizer.save(model_path)
    print(f"人脸特征模型训练完成，并保存到{model_path}文件中！")

    # # 对第一张图像进行人脸识别和匹配
    # label, confidence = recognizer.predict(faces[0])
    # print('name         ', labels_test.get(str(label), "Unknown"))
    # print('confidence   ', str(confidence))
    # if confidence < 50:
    #     print('匹配成功！')
    #遍历识别图片数据对象
    for i, face in enumerate(faces):
        label, confidence = recognizer.predict(face)
        print(f'Image {i + 1}:')
        print('name         ', labels_test.get(str(label), "Unknown"))
        print('confidence   ', str(confidence))
        if confidence < 50:
            print('匹配成功！')
        print('------------------------')
else:
    print("训练失败")

