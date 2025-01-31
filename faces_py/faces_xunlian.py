import cv2
import os
import numpy as np

# 初始化两个列表，用于存储人脸图像数据和对应的标签
faces = []
labels = []

# 定义一个字典，用于存储标签的映射关系
labels_test = {"0": "mark"}

# 遍历"faces"目录下的所有文件
for f in os.listdir("faces"):
    # 如果文件不是jpg格式的图像文件，则跳过
    if '.jpg' not in f:
        continue

    # 使用cv2.imread()函数读取图像文件，并将其转换为灰度图像
    # cv2.IMREAD_GRAYSCALE表示以灰度模式读取图像
    face_image = cv2.imread("faces\\" + f, cv2.IMREAD_GRAYSCALE)

    # 将读取到的灰度人脸图像添加到faces列表中
    faces.append(face_image)

    # 将标签0添加到labels列表中，表示这些人脸图像都属于标签0对应的类别
    labels.append(0)

# 如果faces列表中至少有一个元素（即至少读取到一张人脸图像）
if len(faces) > 0:
    # 创建一个LBPHFaceRecognizer对象，用于人脸识别
    # LBPHFaceRecognizer是一种基于局部二值模式直方图的人脸识别算法
    capture = cv2.face.LBPHFaceRecognizer.create()

    # 使用capture.train()方法训练人脸识别模型
    # 输入参数为faces列表（人脸图像数据）和labels列表（对应的标签）
    capture.train(faces, np.array(labels))

    # 使用capture.save()方法将训练好的模型保存到"face_model.yml"文件中
    # 在以后的程序中加载并使用这个模型进行人脸识别
    capture.save("face_model.yml")

    # 输出提示信息，表示人脸特征模型训练完成，并保存到.yml文件中
    print("人脸特征模型训练完成，并保存到face_model.yml文件中！")