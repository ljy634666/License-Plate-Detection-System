import cv2
import numpy as np
import matplotlib.pyplot as plt

def compare_histograms(original_image, processed_image):
    # 计算直方图
    original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    processed_hist = cv2.calcHist([processed_image], [0], None, [256], [0, 256])

    # 归一化直方图
    original_hist /= original_image.size
    processed_hist /= processed_image.size

    # 计算直方图差异
    correlation = cv2.compareHist(original_hist, processed_hist, cv2.HISTCMP_CORREL)

    return correlation

def main():
    # 读取图像
    image = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)

    # 灰度化
    gray_image = image

    # 高斯去噪
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 中值滤波
    median_blur = cv2.medianBlur(gaussian_blur, 5)

    # 比较直方图
    correlation = compare_histograms(gray_image, median_blur)
    print(f"Histogram Correlation: {correlation}")

    # 显示图像和直方图
    plt.subplot(2, 2, 1), plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 2), plt.imshow(gaussian_blur, cmap='gray')
    plt.title('Gaussian Blur'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 2, 3), plt.imshow(median_blur, cmap='gray')
    plt.title('Median Blur'), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == "__main__":
    main()