import os.path

import matplotlib.pyplot as plt
import numpy as np
import re
import cv2 as cv
import pyqtgraph as pg  # 用于可视化数据
from PyQt5.QtWidgets import QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QSlider, QHBoxLayout, QSpinBox, \
    QLabel, QTabWidget, QGroupBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from algorithm import detect_plate, recognize_characters, \
 \
    segment_characters_canny, detect_license_plate_color, colorRecognition, display_characters, segment_foreground
# 导入algorithm.py中的函数

from pylab import mpl

# 设置中文字体，确保可以正常显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

class LicensePlateDetector(QMainWindow):  # 定义主窗口类，继承QMainWindo
    def __init__(self):
        super().__init__()
        self.initUI()
        self.showMaximized()

    def initUI(self):
        layout = QVBoxLayout()  # 创建垂直布局
        self.tabWidget = QTabWidget()  # 创建标签页控件

        # 创建ImageView控件以显示各种图像
        self.viewOriginal = pg.ImageView()
        self.viewGray = pg.ImageView()
        self.viewSobel = pg.ImageView()
        self.viewBinary = pg.ImageView()
        self.viewMorph = pg.ImageView()
        self.viewPlate = pg.ImageView()
        self.viewChars = pg.ImageView()

        # 添加标签页，并设置标题
        self.tabWidget.addTab(self.viewOriginal, "原图")
        self.tabWidget.addTab(self.viewGray, "灰度图")
        self.tabWidget.addTab(self.viewSobel, "Sobel")
        self.tabWidget.addTab(self.viewBinary, "二值图")
        self.tabWidget.addTab(self.viewMorph, "形态学")
        self.tabWidget.addTab(self.viewPlate, "车牌检测")
        self.tabWidget.addTab(self.viewChars, "字符分割")
        # 将标签页控件添加到垂直布局
        layout.addWidget(self.tabWidget)

        btnLayout = QHBoxLayout()  # 创建水平布局用于按钮

        btnFont = QFont()
        btnFont.setPointSize(12)  # 调整到适中的字体大小

        self.loadButton = QPushButton("加载图片", self)
        self.loadButton.setFont(btnFont)
        self.loadButton.clicked.connect(self.loadImage)
        btnLayout.addWidget(self.loadButton)

        self.detectButton = QPushButton("开始检测", self)
        self.detectButton.setFont(btnFont)
        self.detectButton.clicked.connect(self.detectLicensePlate)
        btnLayout.addWidget(self.detectButton)

        # Parameter Panel
        sobelGroup = QGroupBox("Sobel 参数")
        sobelLayout = QHBoxLayout()
        self.ksizeSpinBox = QSpinBox()
        self.ksizeSpinBox.setRange(1, 10)
        self.ksizeSpinBox.setSingleStep(2)
        self.ksizeSpinBox.setValue(3)
        sobelLayout.addWidget(QLabel("ksize:"))
        sobelLayout.addWidget(self.ksizeSpinBox)
        sobelGroup.setLayout(sobelLayout)

        # Character Recognition
        self.charRecognitionLabel = QLabel("字符识别及成功概率:")
        btnLayout.addWidget(self.charRecognitionLabel)
        self.charRecognitionResult = QLabel("")
        btnLayout.addWidget(self.charRecognitionResult)

        # Vehicle Color
        self.vehicleColorLabel = QLabel("车身颜色:")
        btnLayout.addWidget(self.vehicleColorLabel)
        self.vehicleColorResult = QLabel("")
        btnLayout.addWidget(self.vehicleColorResult)

        # License Plate Color
        self.plateColorLabel = QLabel("车牌颜色:")
        btnLayout.addWidget(self.plateColorLabel)
        self.plateColorResult = QLabel("")
        btnLayout.addWidget(self.plateColorResult)

        morphGroup = QGroupBox("形态学 参数")
        morphLayout = QHBoxLayout()
        self.rectWidthSpinBox = QSpinBox()
        self.rectWidthSpinBox.setRange(1, 50)
        self.rectWidthSpinBox.setValue(17)
        morphLayout.addWidget(QLabel("矩形宽度:"))
        morphLayout.addWidget(self.rectWidthSpinBox)

        self.rectHeightSpinBox = QSpinBox()
        self.rectHeightSpinBox.setRange(1, 50)
        self.rectHeightSpinBox.setValue(3)
        morphLayout.addWidget(QLabel("矩形高度:"))
        morphLayout.addWidget(self.rectHeightSpinBox)
        morphGroup.setLayout(morphLayout)

        cannyGroup = QGroupBox("Canny 参数")
        cannyLayout = QHBoxLayout()
        self.cannyLowSpinBox = QSpinBox()
        self.cannyLowSpinBox.setRange(50, 200)
        self.cannyLowSpinBox.setValue(100)
        cannyLayout.addWidget(QLabel("低阈值:"))
        cannyLayout.addWidget(self.cannyLowSpinBox)

        self.cannyHighSpinBox = QSpinBox()
        self.cannyHighSpinBox.setRange(200, 400)
        self.cannyHighSpinBox.setValue(300)
        cannyLayout.addWidget(QLabel("高阈值:"))
        cannyLayout.addWidget(self.cannyHighSpinBox)
        cannyGroup.setLayout(cannyLayout)

        btnLayout.addWidget(sobelGroup)
        btnLayout.addWidget(morphGroup)
        btnLayout.addWidget(cannyGroup)

        layout.addLayout(btnLayout)  # 将按钮布局添加到垂直布局

        centralWidget = QWidget()  # 创建中央窗口小部件
        centralWidget.setLayout(layout)  # 设置垂直布局作为中央窗口小部件的布局
        self.setCentralWidget(centralWidget)  # 将中央窗口小部件设置为主窗口的中央窗口小部件

        self.setWindowTitle("车牌检测器")  # 设置主窗口标题

    def loadImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", ".", "图片文件(*.png *.jpg *.jpeg)")
        if fname:
            print(f"选择的文件路径: {fname}")
            self.image = cv.imread(fname)
            if self.image is not None:
                # 将图像转换为RGB格式
                self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                print("图像加载成功。")
                self.viewOriginal.setImage(self.image.swapaxes(0, 1))
            else:
                print("无法加载图像。")

    def detectLicensePlate(self):
        # 从界面控件获取参数值
        ksize = self.ksizeSpinBox.value()
        rectWidth = self.rectWidthSpinBox.value()
        rectHeight = self.rectHeightSpinBox.value()
        cannyLow = self.cannyLowSpinBox.value()
        cannyHigh = self.cannyHighSpinBox.value()

        # 复制原始图像
        original_image = self.image.copy()

        # 初始值
        images = detect_plate(original_image, ksize, rectWidth, rectHeight, cannyLow, cannyHigh)

        # 更新ImageView控件以显示处理过的图像
        self.viewGray.setImage(images['gray'].swapaxes(0, 1))
        self.viewSobel.setImage(images['sobel'].swapaxes(0, 1))
        self.viewBinary.setImage(images['binary'].swapaxes(0, 1))
        self.viewMorph.setImage(images['morph'].swapaxes(0, 1))
        self.viewPlate.setImage(images['detected'].swapaxes(0, 1))

        char_segmentation_image = images['detected'].copy()

        # cv.imshow('char_segmentation_image', char_segmentation_image)

        # 对车牌区域进行字符识别
        characters_info = images['characters_info']
        # 检查是否检测到了车牌
        if characters_info:
            # 对车牌区域进行字符识别
            characters = recognize_characters([char_info['image'] for char_info in characters_info],
                                              images['original_gray'])

            # 提取字符文本并去除概率值
            extracted_text = " ".join([char_info['text'].split('（')[0] for char_info in characters])

            # 设置提取后的文本到界面
            self.charRecognitionResult.setText(extracted_text.strip())

            # 在原图上标记每个字符区域
            for char_info in characters_info:
                char_coordinates = char_info['coordinates']
                cv.rectangle(images['detected'], (char_coordinates[0], char_coordinates[1]),
                             (char_coordinates[2], char_coordinates[3]), (0, 255, 0), 2)

            # 在viewChars中显示每个字符的图像
            char_images = [char_info['image'] for char_info in characters_info]
            char_images_swapped = [char_images.swapaxes(0, 1) for char_images in char_images]
            self.viewChars.setImage(char_images_swapped[0])  # 这里只显示第一个字符的图像，可以根据需要修改
            # 字符分割
            result_image = display_characters(char_images_swapped[0], segment_characters_canny(char_images_swapped[0]))
            self.viewChars.setImage(result_image)
        else:
            # 如果未检测到车牌，可以添加相应的处理逻辑
            self.charRecognitionResult.setText("未检测到车牌")


        #处理颜色
        #foreground=segment_foreground(original_image)
        #cv.imshow('char_segmentation_image', foreground)
        data = colorRecognition(original_image)
        result_color = detect_license_plate_color(char_images_swapped[0])

        #车身颜色
        self.vehicleColorResult.setText(''.join(data[0]))
        #车牌颜色
        self.plateColorResult.setText(result_color)


        #fenge = process_and_segment_image(char_images_swapped[0])
        #self.viewChars.setImage(fenge)