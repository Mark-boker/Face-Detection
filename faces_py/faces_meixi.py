import cv2

# 读灰度图
# img = cv2.imread('meixi/meixi01.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('meixi/meixi01.jpeg')

# 检测分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faces = face_cascade.detectMultiScale(img)
print(len(faces))
print(faces)

for (x, y, w, h,) in faces:
    cv2.rectangle(img, (x, y), (x + h, y + w), color=(0, 255, 0))
cv2.imshow('pic', img)


# height, width, channels = img.shape
# print("宽:", width, "高：", height)
# img2 = cv2.resize(img, (int(width / 2), int(height / 2)))
# cv2.imshow('pic', img)
# cv2.imshow('pic2', img2)
cv2.waitKey(0)
cv2.destroyWindow()
