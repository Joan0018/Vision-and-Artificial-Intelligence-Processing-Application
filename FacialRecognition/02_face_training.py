# sudo pip install opencv-contrib-python --break-system-package
# (If needed, use this command to install OpenCV contrib modules / 如果需要，使用此命令安装 OpenCV 扩展模块)

''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                        
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18   

-----------------------------------------------------------------------------------------
训练存储在数据库（dataset 文件夹）中的多个人脸模型：
	==> 每张人脸都应该有一个唯一的数字整数 ID，例如 1, 2, 3 等
	==> 计算出的 LBPH 模型将保存在 trainer/ 目录下。（如果该目录不存在，请手动创建一个）
	==> 为了使用 PIL，请通过 "pip install pillow" 安装 pillow 库
'''

import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database / 人脸图像数据库的路径
path = 'dataset'

# Create LBPH Face Recognizer / 创建 LBPH 人脸识别器对象
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load Haar Cascade Classifier for face detection / 加载用于人脸检测的 Haar 级联分类器
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# function to get the images and label data / 获取图像和标签数据的函数
def getImagesAndLabels(path):

    # Get all file paths in the dataset directory / 获取数据集目录中的所有文件路径
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    
    # Initialize list for face samples / 初始化存放人脸样本的列表
    faceSamples=[]
    
    # Initialize list for user IDs / 初始化存放用户 ID 的列表
    ids = []

    # Loop through all image paths / 遍历所有图像路径
    for imagePath in imagePaths:

        # Open image and convert it to grayscale / 打开图像并将其转换为灰度图
        PIL_img = Image.open(imagePath).convert('L') 
        
        # Convert PIL image to numpy array (uint8 format) / 将 PIL 图像转换为 numpy 数组（uint8 格式）
        img_numpy = np.array(PIL_img,'uint8')

        # Print current image path / 打印当前正在处理的图像路径
        print(imagePath)
        
        # Extract the user ID from the file name (e.g., dataset/User.1.3.jpg -> extract 1)
        # 从文件名中提取用户 ID（例如：dataset/User.1.3.jpg -> 提取出中间的数字 1）
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        
        # Detect faces in the current image / 在当前图像中检测人脸
        faces = detector.detectMultiScale(img_numpy)

        # Loop through detected faces / 遍历检测到的人脸区域
        for (x,y,w,h) in faces:
            # Crop the face region and add to samples / 裁剪人脸区域并添加到样本列表中
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            # Append the corresponding ID / 将对应的 ID 添加到列表中
            ids.append(id)

    # Return face samples and IDs / 返回所有人脸样本和对应的 ID 列表
    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# [提示] 正在训练人脸模型。这需要几秒钟，请稍候...

# Get faces and IDs / 获取人脸数据和 ID
faces,ids = getImagesAndLabels(path)

# Train the model with face samples and IDs / 使用人脸样本和相应的 ID 训练模型
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml / 将训练好的模型保存到 trainer/trainer.yml
# recognizer.save() worked on Mac, but not on Pi / recognizer.save() 在 Mac 上可行，但在树莓派(Pi)上不行
recognizer.write('trainer/trainer.yml') 

# Print the number of faces trained and end program / 打印训练好的人脸数量并结束程序
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
# [提示] 已成功训练 {0} 个不同用户的人脸。正在退出程序。