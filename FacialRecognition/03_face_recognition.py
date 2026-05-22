'''
Real Time Face Recognition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                        
	==> LBPH computed model (trained faces) should be on trainer/ dir

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

-----------------------------------------------------------------------------------------
实时人脸识别：
	==> 存储在 dataset/ 目录中的每张人脸都应该有一个唯一的数字整数 ID（如 1, 2, 3 等）
	==> 计算出的 LBPH 模型（已训练的人脸数据）应该放在 trainer/ 目录下
'''
import cv2
import numpy as np
import os 

# Create LBPH Face Recognizer / 创建 LBPH 人脸识别器对象
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model / 加载训练好的模型文件
recognizer.read('trainer/trainer.yml')

# Path to Haar Cascade XML file / Haar 级联分类器 XML 文件的路径
cascadePath = "haarcascade_frontalface_default.xml"

# Load the face cascade classifier / 加载人脸级联分类器
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set the font type for text rendering on screen / 设置在屏幕上渲染文本的字体类型
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize ID counter / 初始化 ID 计数器
id = 0

# List of names associated with IDs: example ==> Marcelo: id=1, etc
# 与各个 ID 关联的姓名列表：例如 ==> Marcelo 的 ID 是 1，以此类推
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'wy', 'W', 'Joan','123','qr','Hasrul','Andy'] 

# Assign a new value to index 10 / 为索引为 10 的位置重新赋值
names[10] = ['wenyeeeeeeee'] 

# Initialize and start realtime video capture / 初始化并启动实时视频捕获（打开摄像头）
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width / 设置视频宽度为 640
cam.set(4, 480) # set video height / 设置视频高度为 480

# Define minimum window size to be recognized as a face
# 定义能被识别为人脸的最小窗口尺寸（低于该尺寸的区域将被忽略）
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    # Read a single frame from the camera / 从摄像头读取一帧画面
    ret, img = cam.read()
    
    # img = cv2.flip(img, -1) # Flip vertically / 如果画面颠倒，取消注释以进行垂直翻转

    # Convert the frame to grayscale / 将画面转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image / 在灰度图像中检测人脸
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    # Loop through all detected faces / 遍历检测到的所有人脸区域
    for(x,y,w,h) in faces:

        # Draw a green rectangle around the face / 在人脸周围画一个绿色的矩形框（线条粗细为 2）
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Predict the face ID and confidence / 预测人脸的 ID 和置信度（这里的置信度实际是 LBPH 的距离值）
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match 
        # 检查置信度（距离）是否小于 100 ==> “0” 表示完美匹配
        if (confidence < 100):
            # id = names[id] # (Commented out in original) Map ID to name / （原代码中已注释）将数字 ID 映射为对应的名字
            print(id) # Print ID to console / 将 ID 打印到控制台
            
            # Calculate similarity percentage (lower distance means higher confidence)
            # 计算相似度百分比（距离越小，匹配度越高）
            confidence = "  {0}%".format(round(100 - confidence))

        # (Commented out block for handling unknown faces)
        # （用于处理未知人脸的注释代码块）
        # else:
        #     id = "unknown"
        #     confidence = "  {0}%".format(round(100 - confidence))
        #     cam.release()
        #     cv2.destroyAllWindows()
        #     exit()
            
        # Draw the ID text above the rectangle / 在矩形框上方绘制 ID 文本
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        
        # Draw the confidence percentage below the rectangle / 在矩形框下方绘制置信度百分比
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    # Display the real-time video frame / 显示实时视频画面窗口
    cv2.imshow('camera',img) 

    # Wait for 30ms and check if 'ESC' key is pressed
    # 等待 30 毫秒，并检查是否按下了 'ESC' 键（ASCII 码为 27）
    k = cv2.waitKey(30) & 0xff 
    if k == 27: # Press 'ESC' for exiting video / 按下 'ESC' 键退出视频
        break

# Do a bit of cleanup / 退出程序前的清理工作
print("\n [INFO] Exiting Program and cleanup stuff") # [提示] 正在退出程序并清理资源
cam.release() # Release the camera / 释放摄像头
cv2.destroyAllWindows() # Close all OpenCV windows / 关闭所有 OpenCV 窗口