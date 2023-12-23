# 定义检测车牌的函数
import cv2
import numpy as np
import pytesseract
import os
import re
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

pytesseract.pytesseract.tesseract_cmd = r'F:\\tesseract(opencv)\\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = 'F:\\tesseract(opencv)\\tessdata'


def findPlateNumberRegion(img):
    # 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓
    max_contour = None
    max_area = 0

    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓的面积
        area = cv2.contourArea(cnt)

        # 如果当前轮廓面积大于最大面积，则更新最大轮廓
        if area > max_area:
            max_area = area
            max_contour = cnt

    # 如果找到了最大轮廓，将其转换为矩形框返回
    if max_contour is not None:
        epsilon = 0.001 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 车牌正常情况下长高比在2.7-5之间
        ratio = float(width) / float(height)

        # 添加高宽比条件
        if 2 < ratio < 5:
            return [box]

    # 如果未找到车牌区域，返回空列表或其他适当的值
    return []



def detect_plate(image, ksize, rectWidth, rectHeight, cannyLow, cannyHigh):
    results = {}  # 创建一个空字典，用于存储处理后的图像和其他信息

    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results['gray'] = gray

    # 保存原始的灰度图像
    original_gray = gray.copy()

    # 高斯平滑
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)

    # 中值滤波
    median = cv2.medianBlur(gaussian, 5)
    # Sobel算子，X方向求梯度
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=ksize)
    results['sobel'] = sobel

    # cv.imshow('dilation2', sobel)
    # cv.waitKey(0)

    # 二值化
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
    results['binary'] = binary

    # cv.imshow('dilation2', binary)
    # cv.waitKey(0)

    # 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    #dilation2 = cv2.erode(dilation2, element1, iterations=3)

    # cv.imshow('dilation2', dilation2)
    # cv.waitKey(0)

    # 形态学结果
    results['morph'] = dilation2

    # 轮廓检测
    # 查找车牌区域

    region = findPlateNumberRegion(dilation2)

    characters_info = []
    for box in region:
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
    ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
    xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
    ys_sorted_index = np.argsort(ys)
    xs_sorted_index = np.argsort(xs)

    x1 = box[xs_sorted_index[0], 0]
    x2 = box[xs_sorted_index[3], 0]

    y1 = box[ys_sorted_index[0], 1]
    y2 = box[ys_sorted_index[3], 1]
    img_org2 = image.copy()
    img_plate = img_org2[y1:y2, x1:x2]

    characters_info.append({'image': img_plate, 'coordinates': (x1, y1, x2, y2)})

    results['detected'] = image
    results['characters_info'] = characters_info
    results['original_gray'] = original_gray  # 保存原始灰度图像
    return results  # 返回包含结果的字典


#使用pytesseract方法进行字符识别
# def recognize_characters(character_images, original_gray):
#     recognized_text = []
#
#     for character_image in character_images:
#         text = ""
#         attempt_count = 0
#
#         while not text.strip() and attempt_count < 3:
#             # 尝试使用 pytesseract 进行字符识别
#             text = pytesseract.image_to_string(original_gray, lang='chi_sim', config='--psm 6')
#             attempt_count += 1
#
#         recognized_text.append({'text': text.strip()})
#
#     return recognized_text

#使用百度ocr方法进行字符识别
def recognize_characters(character_images, original_gray):
    recognized_text = []

    # 初始化 PaddleOCR
    ocr = PaddleOCR()

    for character_image in character_images:
        # 尝试使用 PaddleOCR 进行字符识别
        result = ocr.ocr(character_image, cls=True)

        # 解析识别结果
        text = ""
        for line in result:
            for word_info in line:
                # 使用索引 -1 获取最后一个元素，并确保它是字符串
                char_str = str(word_info[-1])
                text += char_str

        # 使用正则表达式提取字符
        match = re.search(r'text:\s*(\S+)', text)
        if match:
            extracted_text = match.group(1)
            recognized_text.append({'text': extracted_text.strip()})
        else:
            recognized_text.append({'text': text.strip()})

    return recognized_text


# def satisfy_condition(characters):
#     # 将字符列表连接成一个字符串
#     char_str = ''.join(char_info['text'] for char_info in characters)
#
#     # 使用新的正则表达式进行匹配
#     pattern = re.compile(
#         r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1}$')
#     match = pattern.match(char_str)
#
#     # 如果匹配成功，则满足条件
#     return bool(match)

 #投影法字符分割
# def segment_characters(plate_image):
#     # 将车牌图像转换为灰度图
#     gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
#
#     # 应用二值化
#     _, binary_plate = cv2.threshold(gray_plate, 128, 255, cv2.THRESH_BINARY_INV)
#
#     # 获取图像的投影
#     horizontal_projection = np.sum(binary_plate, axis=1)
#
#     # 设置投影阈值，根据投影的高度来确定字符之间的分割线
#     threshold = binary_plate.shape[1] * 255 * 0.5
#
#     # 根据水平投影的高度，确定字符的上下边界
#     start = 0
#     segments = []
#
#     for i, projection_value in enumerate(horizontal_projection):
#         if projection_value > threshold and start == 0:
#             start = i
#         elif projection_value <= threshold and start != 0:
#             end = i
#             # 添加字符图像及坐标
#             character = gray_plate[start:end, :]
#             segments.append({'image': character, 'coordinates': (0, start, binary_plate.shape[1], end)})
#             start = 0
#
#     return segments


def segment_characters_canny():
    return None

# def process_and_segment_image(img1):
#     img = cv2.resize(img1, (488, 145), interpolation=cv2.INTER_AREA)
#     # cv2.imshow('image1', img)
#
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # cv2.imshow('image2', img_gray)
#
#     img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
#     # cv2.imshow('image3', img_blur)
#
#     _, thresh1 = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY)
#
#     height, width = thresh1.shape[:2]
#     white_max, black_max = 0, 0
#     white, black = [], []
#
#     for i in range(width):
#         s, t = 0, 0
#         for j in range(height):
#             if thresh1[j][i] == 255:
#                 s += 1
#             if thresh1[j][i] == 0:
#                 t += 1
#         white_max = max(white_max, s)
#         black_max = max(black_max, t)
#         white.append(s)
#         black.append(t)
#
#     arg = black_max > white_max
#
#     def inverse_color(edged):
#         height, width = edged.shape
#         img2 = edged.copy()
#         for i in range(height):
#             for j in range(width):
#                 img2[i, j] = (255 - edged[i, j])
#         return img2
#
#     if arg:
#         thresh1 = inverse_color(thresh1)
#     # cv2.imshow('image5', thresh1)
#
#     (h, w) = thresh1.shape
#     a = [0 for z in range(0, w)]
#     black_max1, white_max1 = 0, 0
#
#     for j in range(0, w):
#         for i in range(0, h):
#             if thresh1[i, j] == 0:
#                 a[j] += 1
#                 black_max1 += 1
#                 thresh1[i, j] = 255
#             else:
#                 white_max1 += 1
#
#     for j in range(0, w):
#         for i in range(h - a[j], h):
#             thresh1[i, j] = 0
#
#     # cv2.imshow('image6', thresh1)
#
#     image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     blue = (255, 0, 0)
#
#     def find_end(start_):
#         end_ = start_ + 1
#         for m in range(start_ + 1, width - 1):
#             if (black[m] if arg else white[m]) > (0.95 * black_max if arg else 0.95 * white_max):
#                 end_ = m
#                 break
#         return end_
#
#     n, start, end = 1, 1, 2
#
#     while n < width - 2:
#         n += 1
#         if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
#             start = n
#             end = find_end(start)
#             n = end
#             if end - start > 5:
#                 cj = image[1:height, start:end]
#                 # cv2.imshow('image7', cj)
#                 image7 = cv2.rectangle(image, (start, 10), (end, 140), blue, 2)
#                 cv2.waitKey(0)
#
#     cv2.imshow('image8', image7)
#     cv2.waitKey(0)
#

#分割图像前景
def segment_foreground(img):

    # 创建掩码
    mask = np.zeros(img.shape[:2], np.uint8)

    # 定义矩形ROI
    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)

    # 初始化模型
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # 运行 grabCut 算法
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 更新掩码
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 将原始图像和分割结果相乘，保留前景
    result = img * mask2[:, :, np.newaxis]
    result[mask2 == 0] = [0, 0, 0]

    return result

#车身颜色检测
def colorRecognition(imges):
    #img2 = imges[imges.shape[0] * 1 // 5:imges.shape[0] * 4 // 5, imges.shape[1] * 1 // 5:imges.shape[1] * 4 // 5]
    w, h, c = imges.shape
    w_2 = w // 5
    h_2 = h // 5
    total = w_2 * h_2
    HSV_list = []
    rata_rlist = []
    rata_ylist = []
    rata_blist = []
    rata_glist = []
    rata_bllist = []
    imges1=cv2.cvtColor(imges, cv2.COLOR_BGR2RGB)
    HSV_total = cv2.cvtColor(imges1, cv2.COLOR_BGR2HSV)
    for i in range(5):
        for j in range(5):
            HSV_list.append(HSV_total[i * w_2:(i + 1) * w_2, j * h_2:(j + 1) * h_2])

    # 红色掩膜
    for HSV in HSV_list:
        lower_r1 = np.array([0, 46, 43])
        upper_r1 = np.array([20, 256, 256])
        mask_r1 = cv2.inRange(HSV, lower_r1, upper_r1)
        lower_r1_ref = np.array([0, 0, 225])
        upper_r1_ref = np.array([8, 46, 256])
        mask_r1_ref = cv2.inRange(HSV, lower_r1_ref, upper_r1_ref)
        lower_r2 = np.array([140, 46, 43])
        upper_r2 = np.array([181, 256, 256])
        mask_r2 = cv2.inRange(HSV, lower_r2, upper_r2)
        lower_r2_ref = np.array([140, 0, 225])
        upper_r2_ref = np.array([181, 46, 256])
        mask_r2_ref = cv2.inRange(HSV, lower_r2_ref, upper_r2_ref)
        mask_r = mask_r1 + mask_r2 + mask_r1_ref + mask_r2_ref
        rata_r = np.sum(mask_r == 255) / total
        rata_rlist.append(rata_r)

        # 白色掩膜
        lower_y1 = np.array([0, 0, 100])
        upper_y1 = np.array([180, 30, 255])
        mask_y1 = cv2.inRange(HSV, lower_y1, upper_y1)
        lower_y1_ref = np.array([0, 0, 128])
        upper_y1_ref = np.array([180,70, 255])
        mask_y1_ref = cv2.inRange(HSV, lower_y1_ref, upper_y1_ref)
        mask_y = mask_y1 + mask_y1_ref
        rata_y = np.sum(mask_y == 255) / total
        rata_ylist.append(rata_y)

        # 蓝色掩膜
        lower_b1 = np.array([93, 70, 43])
        upper_b1 = np.array([110, 256, 256])
        mask_b1 = cv2.inRange(HSV, lower_b1, upper_b1)
        lower_b1_ref = np.array([93, 70, 225])
        upper_b1_ref = np.array([110, 70, 256])
        mask_b1_ref = cv2.inRange(HSV, lower_b1_ref, upper_b1_ref)
        mask_b = mask_b1 + mask_b1_ref
        rata_b = np.sum(mask_b == 255) / total
        rata_blist.append(rata_b)

        # 绿色掩膜
        lower_g1 = np.array([70, 46, 43])
        upper_g1 = np.array([93, 256, 256])
        mask_g1 = cv2.inRange(HSV, lower_g1, upper_g1)
        lower_g1_ref = np.array([70, 0, 225])
        upper_g1_ref = np.array([93, 46, 256])
        mask_g1_ref = cv2.inRange(HSV, lower_g1_ref, upper_g1_ref)
        mask_g = mask_g1 + mask_g1_ref
        rata_g = np.sum(mask_g == 255) / total
        rata_glist.append(rata_g)

        # 黑色掩膜
        lower_bl = np.array([0, 0, 0])
        upper_bl = np.array([180, 255, 70])
        mask_bl = cv2.inRange(HSV, lower_bl, upper_bl)
        rata_bl = np.sum(mask_bl == 255) / total
        rata_bllist.append(rata_bl)

    # 进行颜色区域计数，颜色占比超过50%则判定该区域为该颜色。
    num_r = np.sum(np.array(rata_rlist) > 0.65)
    num_y = np.sum(np.array(rata_ylist) > 0.65)
    num_g = np.sum(np.array(rata_glist) > 0.65)
    num_b = np.sum(np.array(rata_blist) > 0.65)
    num_bl = np.sum(np.array(rata_bllist) > 0.65)

    color_num = [num_r, num_y, num_g, num_b, num_bl]
    color_names = ['红色', '白色', '绿色', '蓝色', '黑色']
    color_rgb = ['255,0,0', '255,255,255', '0,255,0', '0,0,255', '0,0,0']

    score = max(color_num)
    if score < 5:
        color_result = '白色'
        color_rgb_result = '255,255,255'
        color_proportion_result = score
    else:
        max_location = color_num.index(score)
        color_result = color_names[max_location]
        color_rgb_result = color_rgb[max_location]
        color_proportion_result = score
    data = [color_result, color_proportion_result, color_rgb_result]
    return data




#车牌颜色检测
def detect_license_plate_color(img):

    # 设定阈值
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    lower_yellow = np.array([15, 55, 55])
    upper_yellow = np.array([50, 255, 255])
    lower_green = np.array([0, 3, 116])
    upper_green = np.array([76, 211, 255])

    # 转换为HSV
    hsv1=  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(hsv1, cv2.COLOR_BGR2HSV)

    # 根据阈值构建掩膜
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 对原图像和掩膜进行位运算
    res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
    res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
    res_green = cv2.bitwise_and(img, img, mask=mask_green)

    # 对mask进行操作--黑白像素点统计
    # 因为不同颜色的掩膜面积不一样
    # 记录黑白像素总和
    blue_white = np.sum(mask_blue == 255)
    blue_black = np.sum(mask_blue == 0)
    yellow_white = np.sum(mask_yellow == 255)
    yellow_black = np.sum(mask_yellow == 0)
    green_white = np.sum(mask_green == 255)
    green_black = np.sum(mask_green == 0)

    color_list = ['蓝色', '黄色', '绿色']
    num_list = [blue_white, yellow_white, green_white]
    result_color = color_list[num_list.index(max(num_list))]

    return result_color
# 边缘检测法字符分割
def segment_characters_canny(plate_image):
    # 将车牌图像进行水平翻转
    plate_image_flipped = cv2.flip(plate_image, 1)

    # 将翻转后的车牌图像转换为灰度图
    gray_plate = cv2.cvtColor(plate_image_flipped, cv2.COLOR_BGR2GRAY)

    # 图像预处理，高斯模糊
    blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)
    #中值滤波
    median = cv2.medianBlur(blurred_plate, 5)

    # 二值化
    _, binary_plate = cv2.threshold(median, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(binary_plate, 30, 150)  # 调整阈值

    # 多次膨胀和腐蚀（开闭运算）
    for _ in range(3):
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 根据轮廓获取字符的位置和图像
    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 忽略过小的轮廓
        min_contour_area = 100  # 设置一个最小的轮廓面积
        min_contour_width = 10  # 设置一个最小的轮廓宽度

        if cv2.contourArea(contour) > min_contour_area and w > min_contour_width:
            # 调整坐标
            x_flipped = plate_image.shape[1] - (x + w)
            segments.append({'image': gray_plate[y:y + h, x:x + w], 'coordinates': (x_flipped, y, x_flipped + w, y + h)})

    return segments

# 在原图中绘制矩形框并显示
# def display_characters(image, characters_info):
#     image_with_rectangles = image.copy()
#
#     for char_info in characters_info:
#         coordinates = char_info['coordinates']
#         cv2.rectangle(image_with_rectangles, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 2)
#
#     # 顺时针旋转90度
#     image_with_rectangles_rotated = np.rot90(image_with_rectangles, k=1)
#
#     # 镜像翻转图像
#     image_with_rectangles_rotated_flipped = np.flipud(image_with_rectangles_rotated)
#
#     # 显示带有矩形框的顺时针旋转90度并镜像翻转的图像，别问为什么要旋转翻转，反正结果是对的
#     plt.imshow(image_with_rectangles_rotated_flipped)
#     plt.title("车牌字符分割")
#     plt.axis('off')
#     plt.show()


def display_characters(image, characters_info):
    image_with_rectangles = image.copy()

    for char_info in characters_info:
        coordinates = char_info['coordinates']
        cv2.rectangle(image_with_rectangles, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), (0, 255, 0), 2)

    # # 顺时针旋转90度
    # image_with_rectangles_rotated = np.rot90(image_with_rectangles, k=1)
    #
    # # 镜像翻转图像
    # image_with_rectangles_rotated_flipped = np.flipud(image_with_rectangles_rotated)

    return image_with_rectangles

# 示例用法

