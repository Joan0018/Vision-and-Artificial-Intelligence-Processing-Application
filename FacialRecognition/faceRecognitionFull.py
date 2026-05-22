import cv2
import os
import json
import numpy as np

# --- Configuration & Setup / 配置与初始化 ---
DATASET_DIR = "dataset" # Folder to save face images / 存放人脸图像的文件夹
TRAINER_DIR = "trainer" # Folder to save model & maps / 存放模型和映射文件的文件夹
USER_MAP_FILE = os.path.join(TRAINER_DIR, "users.json") # JSON file mapping IDs to names / 映射 ID 与姓名的 JSON 文件
MODEL_PATH = os.path.join(TRAINER_DIR, "trainer.yml")  # Path for trained LBPH model / 训练好的 LBPH 模型路径

# Create folders automatically if they don't exist
# 如果文件夹不存在，则自动创建它们
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

# Detect Haar Cascade Path automatically / 自动获取 OpenCV 自带的 Haar 级联分类器路径
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(CASCADE_PATH)

# Check if OpenCV Face module is available
# 检查 OpenCV 的人脸识别模块是否可用
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    # Triggered if opencv-contrib-python is missing / 若缺失扩展包则触发提示
    print("Error: Face module missing. Run this command in your Pi Terminal:")
    print("sudo apt update && sudo apt install python3-opencv")
    exit()

# Load ID-to-Name mapping from JSON / 从 JSON 文件加载 ID 到姓名的映射
def load_user_map():
    if os.path.exists(USER_MAP_FILE):
        with open(USER_MAP_FILE, "r") as f:
            return json.load(f)
    return {}

# Save ID-to-Name mapping to JSON / 将 ID 到姓名的映射保存至 JSON 文件
def save_user_map(user_map):
    with open(USER_MAP_FILE, "w") as f:
        json.dump(user_map, f, indent=4)

# --- Mode 1: Face Registration / 模式 1：人脸注册 ---
def register_face():
    user_map = load_user_map()
    
    # Input and validate User ID / 输入并验证用户 ID
    face_id = input("\nEnter a unique numeric User ID (e.g., 1, 2, 3): ").strip()
    if not face_id.isdigit():
        print("Error: ID must be a number! / 错误：ID 必须是纯数字！")
        return
        
    # Input and validate User Name / 输入并验证用户名
    name = input("Enter User Name: ").strip()
    if not name:
        print("Error: Name cannot be empty! / 错误：姓名不能为空！")
        return

    # Update and save user data / 更新并保存用户数据
    user_map[str(face_id)] = name
    save_user_map(user_map)

    # Initialize Camera Stream (Note: "rtsp://" can be replaced with 0 for local USB cam)
    # 初始化摄像头流（注意：如果是本地 USB 摄像头，可将 "rtsp://" 改为 0）
    cam = cv2.VideoCapture("rtsp://")
    print("\nInitializing camera. Look directly at the lens...")
    print("[提示] 正在初始化摄像头。请直视镜头...")
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame. / 获取画面失败。")
            break

        img = cv2.flip(img, 1)  # Mirror view / 左右镜像翻转画面，符合自拍习惯
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale / 转换为灰度图
        faces = face_detector.detectMultiScale(gray, 1.3, 5) # Detect faces / 检测人脸

        for (x, y, w, h) in faces:
            count += 1
            # Save using native OpenCV instead of Pillow
            # 直接使用 OpenCV 原生函数保存裁剪的人脸，无需使用 Pillow 库
            cv2.imwrite(f"{DATASET_DIR}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])
            # Draw blue boundary box / 绘制蓝色人脸框
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the real-time frame / 显示实时注册窗口
        cv2.imshow("Registering Face - Look at the Camera", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27 or count >= 60:  # Press ESC or wait for 60 samples / 按下 ESC 键或集齐 60 张样本后退出
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Data collection complete: Saved {count} images for {name}.\n")
    print(f"[提示] 数据采集完成：已为 {name} 保存了 {count} 张样本图片。\n")

# --- Mode 2: Model Training / 模式 2：模型训练 ---
def train_model():
    print("\nProcessing dataset. Please wait...")
    print("[提示] 正在处理数据集，请稍候...")
    
    # Filter and grab all valid JPG paths / 筛选并获取目录下所有有效的图像路径
    image_paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]

    if not image_paths:
        print("No images found! Register a face first.\n")
        print("[错误] 未找到图像文件！请先注册一个人脸。\n")
        return

    face_samples = []
    ids = []

    for image_path in image_paths:
        # Fixed: Read directly as grayscale using OpenCV to bypass Pillow library
        # 修正：直接通过 OpenCV 以灰度模式读取图片，彻底摆脱对 Pillow 库的依赖
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        filename = os.path.split(image_path)[-1]
        
        try:
            # Extract User ID from name format "User.ID.Count.jpg"
            # 从文件名格式 "User.ID.Count.jpg" 中提取用户数字 ID
            uid = int(filename.split(".")[1])
            faces = face_detector.detectMultiScale(gray_img)
            
            for (x, y, w, h) in faces:
                face_samples.append(gray_img[y:y+h, x:x+w]) # Append cropped face / 添加裁剪的人脸区域
                ids.append(uid) # Append corresponding user ID / 添加对应的用户 ID
        except Exception:
            continue # Skip corrupted or mismatch files / 自动跳过损坏或不匹配的文件

    if face_samples:
        # Train the model with the compiled lists / 使用提取的数据集训练模型
        recognizer.train(face_samples, np.array(ids))
        recognizer.write(MODEL_PATH) # Save the trained matrix / 保存训练好的矩阵文件
        print(f"Success: Model optimized for {len(np.unique(ids))} unique user(s).\n")
        print(f"[成功] 模型训练完成，已优化 {len(np.unique(ids))} 个用户的特征数据。\n")
    else:
        print("Error: No clear faces could be extracted from the dataset.\n")
        print("[错误] 训练失败：无法从现有的数据集中提取出清晰的人脸。\n")

# --- Mode 3: Real-Time Recognition / 模式 3：实时人脸识别 ---
def recognize_faces():
    if not os.path.exists(MODEL_PATH):
        print("Error: Trainer file missing. Run the training mode first.\n")
        print("[错误] 找不到训练好的模型文件。请先运行模式 2 进行训练。\n")
        return

    recognizer.read(MODEL_PATH) # Load the trained model / 加载训练好的模型
    user_map = load_user_map()  # Load names dict / 加载姓名映射字典

    cam = cv2.VideoCapture(0) # Open default local camera / 打开默认的本地摄像头
    
    # Lower resolution slightly to maximize performance frames-per-second on Raspberry Pi
    # 适当调低分辨率，以最大化提升树莓派（Raspberry Pi）上的实时流畅度与每秒帧数（FPS）
    cam.set(3, 480)
    cam.set(4, 360)

    print("\nStarting system recognition window.")
    print("Click on the video window and press 'ESC' to exit.")
    print("\n[提示] 正在启动实时识别窗口。点击视频窗口后按 'ESC' 键可安全退出。")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        img = cv2.flip(img, 1) # Mirror view / 画面镜像翻转
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to gray / 转换为灰度
        faces = face_detector.detectMultiScale(gray, 1.2, 5) # Detect faces / 检测人脸

        for (x, y, w, h) in faces:
            # Draw green box around detected face / 在检测到的人脸周围画绿色矩形框
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Predict identity (uid) and confidence score (loss value)
            # 预测身份 (uid) 以及置信度分数 (此处的 loss 越小代表越匹配)
            uid, loss = recognizer.predict(gray[y:y+h, x:x+w])

            # Check if loss is below threshold (lower loss = better match)
            # 检查特征差距值是否在安全阈值 100 以内（值越低越匹配）
            if loss < 100:
                name = user_map.get(str(uid), f"User {uid}") # Fetch name from json / 从映射表中提取姓名
                confidence = f"{round(100 - loss)}%"        # Convert loss to accurate match % / 将差距转换为匹配度百分比
            else:
                name = "Unknown" # Mismatch / 未能成功匹配
                confidence = f"{round(100 - loss)}%"

            # Draw text labels on the camera frames / 在画面对应的脸框位置渲染名字与匹配率
            cv2.putText(img, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show output window / 渲染主输出窗口
        cv2.imshow("Real-Time Face Recognition", img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # Escape key / 检测是否按下 ESC 键
            break

    # Resource cleanup / 资源释放与清理
    cam.release()
    cv2.destroyAllWindows()
    print("Recognition system stopped.\n")
    print("[提示] 识别系统已停止。\n")

# --- Main Console Loop / 主控制台循环菜单 ---
if __name__ == "__main__":
    while True:
        print("=== Raspberry Pi Facial Recognition ===")
        print("1. Register New Face (注册新面孔)")
        print("2. Train Dataset Model (训练数据集模型)")
        print("3. Run Face Recognition (启动人脸识别)")
        print("4. Exit Application (退出程序)")
        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            register_face()
        elif choice == "2":
            train_model()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("Exiting application. / 正在退出应用程序。")
            break
        else:
            print("Invalid selection. Try again.\n")
            print("[错误] 无效的选择，请重新输入。\n")