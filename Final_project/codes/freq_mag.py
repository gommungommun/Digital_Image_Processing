import cv2
import numpy as np
import matplotlib.pyplot as plt

def frequency_mag(img):
    # 이미지를 주파수 영역으로 변환
    f = np.fft.fft2(img)

    # 분석을 위해 0 주파수(DC 성분)를 이미지 중앙으로 이동
    fshift = np.fft.fftshift(f)

    # 노이즈 성분의 분포를 알아보기 위해 magnitude를 구함
    magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))

    # 원본 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('origin')
    plt.axis('off')

    # 주파수 스펙트럼 표시
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('frequency')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# 주파수 스펙트럼 알아보기
if __name__ == '__main__':
    img = cv2.imread('barcode3.jpg', 0)
    frequency_mag(img)