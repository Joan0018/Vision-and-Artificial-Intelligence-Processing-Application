''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
	==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
	==> Each face will have a unique numeric integer ID as 1, 2, 3, etc                       

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18    

'''

'''
从多个用户捕获多张人脸图像并存储到数据库（dataset 文件夹）
	==> 人脸图像将存储在目录：dataset/（如果该目录不存在，请手动创建一个）
	==> 每张人脸都将有一个唯一的数字整数 ID，如 1, 2, 3 等

基于 Anirban Kar 的原始代码：https://github.com/thecodacus/Face-Recognition  
由 Marcelo Rovai 开发 - MJRoBot.org @ 21Feb18  
'''

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width # 设置视频宽度为 640
cam.set(4, 480) # set video height # 设置视频高度为 480

# 加载人脸检测的 Haar 级联分类器
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
# 为每个人输入一个数字人脸 ID（例如：1, 2, 3...）
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")

# Initialize individual sampling face count
# 初始化个人人脸样本计数器
count = 0

while(True):
    # 读取摄像头的每一帧画面
    ret, img = cam.read()
    
    #img = cv2.flip(img, -1) # flip video image vertically
    # 如果画面颠倒了，可以取消这行的注释来垂直翻转图像
    
    # 将图像转换为灰度图（人脸检测和保存通常使用灰度图以减少计算量）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测画面中的人脸
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        # 在原图的人脸周围画一个蓝色的矩形框（线条粗细为 2）
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        
        count += 1

        # Save the captured image into the datasets folder
        # 将捕获的灰度人脸区域裁剪并保存到 dataset 文件夹中
        # 文件名格式如：User.1.1.jpg, User.1.2.jpg ...
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        # 显示带有矩形框的实时画面
        cv2.imshow('image', img)

    # 等待键盘输入，每帧间隔 30 毫秒
    k = cv2.waitKey(30) & 0xff # Press 'ESC' for exiting video
    if k == 27: # 如果按下 'ESC' 键（ASCII 码为 27），则退出循环
        break
    
    # 成功获取 60 个图像样本后停止视频捕获
    elif count >= 60: # Take 60 face sample and stop video 
         break

# Do a bit of cleanup
# 退出程序前的清理工作
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release() # 释放摄像头
cv2.destroyAllWindows() # 关闭所有 OpenCV 创建的窗口


