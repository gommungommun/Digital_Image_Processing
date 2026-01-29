import cv2
import numpy as np

# zero padding
def zero_padding(img):
    m,n = img.shape
    p,q = m*2, n*2
    padded_img = np.zeros((p,q))
    padded_img[:m, :n] = img

    return padded_img

# 주파수 성분을 위한 로그 연산 - 파라미터로는 제로 패딩된 이미지가 들어와야 함
def dft_log(img):
    m,n = img.shape
    p,q = m*2, n*2
    dft2d_log = np.log(np.abs(img)+1)
    centered_log = np.zeros((p,q))
    # centering
    for x in range(p):
        for y in range(q):
            centered_log[x,y] = dft2d_log[x,y]*((-1)**(x+y))
    
    return centered_log

def GLPF(img, cutoff, parameter):
    m,n = img.shape
    f= np.zeros((m,n))

    U0 = int(m/2)
    V0 = int(n/2)

    D0 = cutoff

    for u in range(m):
        for v in range(n):
            d_uv = np.sqrt((u-U0)**2+(v-V0)**2)
            l, h = parameter
            f[u, v] = (h-l)*(1-np.exp(-d_uv**2/(2*D0**2)))+l

    return f

# barcode 1을 위한 호모모픽 필터


def Laplacian(image):
    M,N = image.shape
    H = np.zeros((M,N))
    