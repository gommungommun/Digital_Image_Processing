"""
Image processing project exam 1
---------------------------------
Name: 손유을
Student ID: 12211060
---------------------------------
Problem 3 Sharpening and Smoothing with Laplacian
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Laplacian(shape):
    M, N = shape
    H = np.zeros((M, N), dtype=np.float32)
    
    U0, V0 = M / 2, N / 2
    
    for u in range(M):
        for v in range(N):
            D_sq = np.power(u - U0, 2) + np.power(v - V0, 2)
            H[u, v] = -D_sq 
            
    return H

def laplacian_fft_image(img):
    M, N = img.shape
    
    P, Q = 2 * M, 2 * N
    padded_img = np.zeros((P, Q))
    padded_img[:M, :N] = img
    
    padded_img_new = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            padded_img_new[x, y] = padded_img[x, y] * ((-1)**(x + y))
            
    dft2d = np.fft.fft2(padded_img_new)
    
    H = Laplacian(dft2d.shape) 
    filtering = dft2d * H      
    
    idft_ = np.fft.ifft2(filtering)
    
    decentering = np.zeros((P, Q), dtype=complex)
    for x in range(P):
        for y in range(Q):
            decentering[x, y] = idft_[x, y] * ((-1)**(x + y))
    
    idft2d_raw = decentering[:M, :N].real
    
    f_min, f_max = np.min(idft2d_raw), np.max(idft2d_raw)
    if f_max - f_min != 0:
        idft2d = (idft2d_raw - f_min) / (f_max - f_min) * 255.0
    else:
        idft2d = idft2d_raw
        
    return idft2d

def Sharpening(img, laplacian_img):
    sharp = cv2.addWeighted(img.astype(float), 0.7, laplacian_img, 0.3, 0)
    return sharp

def Smoothing(img, laplacian_img):
    smooth = cv2.addWeighted(img.astype(float), 1.0, laplacian_img, -0.2, 0)
    return smooth

if __name__ == '__main__':
    img = cv2.imread('Midterm2.jpg', 0)
    if img is None:
        img = np.uint8(np.random.rand(100, 100) * 255)

    laplacian_res = laplacian_fft_image(img)
    sharp = Sharpening(img, laplacian_res)
    smooth = Smoothing(img, laplacian_res)

    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(laplacian_res, cmap='gray')
    plt.title('Laplacian Result')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(sharp, cmap='gray')
    plt.title('Sharpening')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(smooth, cmap='gray')
    plt.title('Smoothing')
    plt.axis('off')

    plt.show()