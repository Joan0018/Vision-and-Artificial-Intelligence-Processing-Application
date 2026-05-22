import cv2
import numpy as np
import math

# LAB color thresholds (from your original script)
# LAB 颜色阈值（源自您原本的脚本）
lower = np.array([94, 44, 139])     # LAB minimum values (Color 1) / LAB 最小值（颜色 1）
upper = np.array([232, 115, 192])   # LAB maximum values (Color 1) / LAB 最大值（颜色 1）

lower2 = np.array([193, 0, 0])      # Alternative LAB minimum values (Color 2) / 备选 LAB 最小值（颜色 2）
upper2 = np.array([206, 119, 202])  # Alternative LAB maximum values (Color 2) / 备选 LAB 最大值（颜色 2）

# CHANGED: 0 targets the default built-in webcam. Try 1 or 2 if you have external USB cams.
# 已修改：0 代表默认的内置摄像头。如果您连接了外置 USB 摄像头，可以尝试修改为 1 或 2。
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set video width / 设置视频帧宽度
cap.set(4, 480)  # Set video height / 设置视频帧高度

# Check if the webcam opened successfully / 检查摄像头是否成功打开
if not cap.isOpened():
    print("Cannot open camera / 无法打开摄像头")
    exit()

while True:
    # Capture frame-by-frame / 逐帧读取视频画面
    ret, img = cap.read()
    
    # img = cv2.flip(img, 1) # Uncomment this if you want a mirror effect for the webcam
    # img = cv2.flip(img, 1) # 如果需要镜像翻转画面（符合自拍视角），请取消本行注释

    # Check if the frame was received correctly / 检查是否成功接收到画面帧
    if not ret:
        print("Cannot receive frame / 无法接收画面帧")
        break

    # Convert image from BGR to LAB color space
    # 将图像从默认的 BGR 格式转换为 LAB 色彩空间
    frame_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Create a binary mask: pixels within the thresholds become white (255), others black (0)
    # 创建二值化掩膜：在阈值范围内的像素变白(255)，其余区域变黑(0)
    frame_mask = cv2.inRange(frame_lab, lower, upper)

    # Morphological operations to filter background noise
    # 执行形态学操作以滤除背景噪点
    
    # 1. Open operation (Erosion followed by Dilation) to remove small white dots (noise)
    # 1. 开运算（先腐蚀后膨胀），用于消除图像中微小的白色噪点
    opened = cv2.morphologyEx(
        frame_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )  
    
    # 2. Close operation (Dilation followed by Erosion) to fill small black holes inside the target
    # 2. 闭运算（先膨胀后腐蚀），用于填补目标物体内部的黑色小孔洞
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )  

    # Find the edges/contours of the isolated shapes in the binary image
    # 在二值化图像中寻找被孤立形状的边缘/轮廓
    contours = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    # Iterate through each detected contour / 遍历检测到的每一个轮廓
    for contour in contours:
        area = cv2.contourArea(contour)  # Get target contour area / 计算当前轮廓的面积
        color = (0, 0, 255)              # Outer bounding box color (BGR format - Red) / 外接矩形框的颜色（BGR 格式 - 红色）

        # Only label and box the target if its area is larger than 300 pixels
        # 仅当轮廓面积大于 300 像素时才进行标记和框选（过滤过小的干扰物）
        if area > 300:
            # Calculate the straight bounding rectangle coordinates / 计算物体的正外接矩形坐标
            x, y, w, h = cv2.boundingRect(contour)  
            
            # Draw the rectangle on the original color image / 在原始彩色图像上绘制矩形框
            img = cv2.rectangle(
                img, (x, y), (x + w, y + h), color, 3
            )  

    # Render and display the processed frame / 渲染并显示处理后的窗口画面
    cv2.imshow("color", img)
    
    # Wait for 30ms and capture key presses / 等待 30 毫秒并捕获键盘输入
    k = cv2.waitKey(30) & 0xff
    if k == 27: # Press 'ESC' key to quit / 如果按下 'ESC' 键（ASCII 码为 27）则退出循环
        break

# Release the camera and destroy all windows / 释放摄像头资源并销毁所有 OpenCV 窗口
cap.release()
cv2.destroyAllWindows()