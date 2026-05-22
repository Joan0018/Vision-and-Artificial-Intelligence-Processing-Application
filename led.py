import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)   #禁用警告
led1_pin = 12
GPIO.setup(led1_pin,GPIO.OUT)  # 将当前管脚设定为输出
while True:                    # 重复  
    GPIO.output(led1_pin, GPIO.LOW)      # 点亮 LED
    time.sleep(0.5)               # 等待 0.5 秒
    GPIO.output(led1_pin, GPIO.HIGH)     # 熄灭 LED
    time.sleep(0.5)               # 等待 0.5 秒