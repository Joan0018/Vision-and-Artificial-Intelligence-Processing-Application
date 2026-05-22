#!/usr/bin/env python3
# encoding: utf-8
import os
import re
import cv2
import sys
import math
import time

from socket import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from color_detect import Ui_Form
from PyQt5 import QtGui, QtWidgets

envpath = '/lib/python3/dist-packages/PyQt5'          #强制设置环境变量,不能删除，否则报错 import cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        #self.setWindowIcon(QIcon(':/logo.png'))
        
        self.color = 'red'
        self.L_Min = 0
        self.A_Min = 0
        self.B_Min = 0
        self.L_Max = 255
        self.A_Max = 255
        self.B_Max = 255
        self.kernel_open = 3
        self.kernel_close = 3
        self.camera_ui = False
        self.camera_opened = False
        self.camera_ui_break = False
        
        self.horizontalSlider_LMin.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('lmin'))
        self.horizontalSlider_AMin.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('amin'))
        self.horizontalSlider_BMin.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('bmin'))
        self.horizontalSlider_LMax.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('lmax'))
        self.horizontalSlider_AMax.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('amax'))
        self.horizontalSlider_BMax.valueChanged.connect(lambda: self.horizontalSlider_labvaluechange('bmax'))

        self.pushButton_connect.pressed.connect(lambda: self.on_pushButton_action_clicked('connect'))
        self.pushButton_disconnect.pressed.connect(lambda: self.on_pushButton_action_clicked('disconnect'))
        self.pushButton_quit.pressed.connect(lambda: self.button_controlaction_clicked('quit'))
        self._timer = QTimer()
        self._timer.timeout.connect(self.show_image)
        self.createConfig()

    ################################################################################################
    #获取面积最大的轮廓
    def getAreaMaxContour(self,contours) :
            contour_area_temp = 0
            contour_area_max = 0
            area_max_contour = None;

            for c in contours :
                contour_area_temp = math.fabs(cv2.contourArea(c)) #计算面积
                if contour_area_temp > contour_area_max : #新面积大于历史最大面积就将新面积设为历史最大面积
                    contour_area_max = contour_area_temp
                    if contour_area_temp > 10: #只有新的历史最大面积大于10,才是有效的最大面积
                                               #就是剔除过小的轮廓
                        area_max_contour = c

            return area_max_contour #返回得到的最大面积，如果没有就是 None
    
    def show_image(self):
        if self.camera_opened:
            ret, orgframe = self.cap.read()
            if ret:
                orgFrame = cv2.resize(orgframe, (400, 300))
                orgFrame = cv2.GaussianBlur(orgFrame, (3, 3), 3)
                frame_lab = cv2.cvtColor(orgFrame, cv2.COLOR_BGR2LAB)
                mask = cv2.inRange(frame_lab,
                                   (self.current_lab_data[self.color]['min'][0],
                                    self.current_lab_data[self.color]['min'][1],
                                    self.current_lab_data[self.color]['min'][2]),
                                   (self.current_lab_data[self.color]['max'][0],
                                    self.current_lab_data[self.color]['max'][1],
                                    self.current_lab_data[self.color]['max'][2]))#对原图像和掩模进行位运算
                opend = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_open, self.kernel_open)))
                closed = cv2.morphologyEx(opend, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_close, self.kernel_close)))
                showImage = QImage(closed.data, closed.shape[1], closed.shape[0], QImage.Format_Indexed8)
                temp_pixmap = QPixmap.fromImage(showImage)
                frame_rgb = cv2.cvtColor(orgFrame, cv2.COLOR_BGR2RGB)
                showframe = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
                t_p = QPixmap.fromImage(showframe)
                
                self.label_process.setPixmap(temp_pixmap)
                self.label_orign.setPixmap(t_p)

    def horizontalSlider_labvaluechange(self, name):
        if name == 'lmin': 
            self.current_lab_data[self.color]['min'][0] = self.horizontalSlider_LMin.value()
            self.label_LMin.setNum(self.current_lab_data[self.color]['min'][0])
        if name == 'amin':
            self.current_lab_data[self.color]['min'][1] = self.horizontalSlider_AMin.value()
            self.label_AMin.setNum(self.current_lab_data[self.color]['min'][1])
        if name == 'bmin':
            self.current_lab_data[self.color]['min'][2] = self.horizontalSlider_BMin.value()
            self.label_BMin.setNum(self.current_lab_data[self.color]['min'][2])
        if name == 'lmax':
            self.current_lab_data[self.color]['max'][0] = self.horizontalSlider_LMax.value()
            self.label_LMax.setNum(self.current_lab_data[self.color]['max'][0])
        if name == 'amax':
            self.current_lab_data[self.color]['max'][1] = self.horizontalSlider_AMax.value()
            self.label_AMax.setNum(self.current_lab_data[self.color]['max'][1])
        if name == 'bmax':
            self.current_lab_data[self.color]['max'][2] = self.horizontalSlider_BMax.value()
            self.label_BMax.setNum(self.current_lab_data[self.color]['max'][2])

    def createConfig(self, c=False):        
        data = {'red': {'max': [255, 255, 255], 'min': [0, 150, 130]},
                'green': {'max': [255, 110, 255], 'min': [47, 0, 135]},
                'blue': {'max': [255, 136, 120], 'min': [0, 0, 0]},
                'black': {'max': [89, 255, 255], 'min': [0, 0, 0]},
                'white': {'max': [255, 255, 255], 'min': [193, 0, 0]}}
        self.current_lab_data = data
        
        self.color_list = ['red', 'green', 'blue', 'black', 'white']
        self.comboBox_color.addItems(self.color_list)
        self.comboBox_color.currentIndexChanged.connect(self.selectionchange)       
        self.selectionchange() 
                                 
    def getColorValue(self, color):
        if color != '':
            self.horizontalSlider_LMin.setValue(self.current_lab_data[color]['min'][0])
            self.horizontalSlider_AMin.setValue(self.current_lab_data[color]['min'][1])
            self.horizontalSlider_BMin.setValue(self.current_lab_data[color]['min'][2])
            self.horizontalSlider_LMax.setValue(self.current_lab_data[color]['max'][0])
            self.horizontalSlider_AMax.setValue(self.current_lab_data[color]['max'][1])
            self.horizontalSlider_BMax.setValue(self.current_lab_data[color]['max'][2])

    def selectionchange(self):
        self.color = self.comboBox_color.currentText()      
        self.getColorValue(self.color)
        
    def on_pushButton_action_clicked(self, buttonName):
        if buttonName == 'connect':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.camera_opened = False
                self.label_process.setText('Can\'t find camera')
                self.label_orign.setText('Can\'t find camera')
                self.label_process.setAlignment(Qt.AlignCenter|Qt.AlignVCenter)
                self.label_orign.setAlignment(Qt.AlignCenter|Qt.AlignVCenter)
            else:
                self.camera_opened = True
                self._timer.start(20)
        elif buttonName == 'disconnect':
            if self.camera_opened:
                self.camera_opened = False
                self._timer.stop()
                self.label_process.setText(' ')
                self.label_orign.setText(' ')
                self.cap.release()
            else:
                self.label_process.setText('Not connected')
                self.label_orign.setText('Not connected')
                
    def button_controlaction_clicked(self, name):
        if name == 'quit':
            self.camera_ui = True
            self.camera_ui_break = True
            try:
                self.cap.release()
            except:
                pass
            sys.exit()
            
            
if __name__ == "__main__":  
    app = QtWidgets.QApplication(sys.argv)
    myshow = MainWindow()
    myshow.show()
    sys.exit(app.exec_())
