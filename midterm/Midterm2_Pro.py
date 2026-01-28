"""
Image processing project exam 1
---------------------------------
Name: 손유을 
Student ID: 12211060
---------------------------------
Problem 2 DFT with Butterworth Low Pass Filter
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Zeropadding(image):
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    padded_image = np.zeros((P, Q))
    padded_image[:M, :N] = image
    return padded_image

def Centering(image):
    P, Q = image.shape
    padded_image_new = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            padded_image_new[x, y] = image[x, y] * ((-1)**(x + y))
    return padded_image_new

def BLPF(image, order=2, cutoff=100):
    M, N = image.shape
    H = np.zeros((M, N))
    U0, V0 = M // 2, N // 2
    D0 = cutoff
    for u in range(M):
        for v in range(N):
            D_uv = np.sqrt((u - U0)**2 + (v - V0)**2)
            H[u, v] = 1 / (1 + (D_uv / D0)**(2 * order))
    return H

def De_centering_and_Remove_Zeropadding(image):
    P, Q = image.shape
    M, N = P // 2, Q // 2
    for x in range(P):
        for y in range(Q):
            image[x, y] = image[x, y] * ((-1)**(x + y))
    return image[:M, :N].real


if __name__ == '__main__':
    img = cv2.imread('Midterm2.jpg', 0)
    if img is None:
        img = np.uint8(np.random.rand(100, 100) * 255) # 파일 없을 시 예외 처리

    zeropadded = Zeropadding(img)
    centered = Centering(zeropadded)
    dft2d = np.fft.fft2(centered)
    
    blpf_filter = BLPF(dft2d, 2, 100)
    filtered_dft = dft2d * blpf_filter
    
    idft2d = np.fft.ifft2(filtered_dft)
    result = De_centering_and_Remove_Zeropadding(idft2d)

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(zeropadded, cmap='gray')
    plt.title('Zero Padding')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(blpf_filter, cmap='gray')
    plt.title('BLPF Filter')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(idft2d.real, cmap='gray')
    plt.title('IDFT (Filtered)')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(result, cmap='gray')
    plt.title('Final Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()