import cv2
import numpy as np
import matplotlib.pyplot as plt

def homorophic(img):
    m,n = img.shape
    p,q = m*2, n*2

    # zero-padding
    padded_img = np.zeros((p,q))
    padded_img[:m, :n] = img

    # log를 취함
    dft2d_log = np.log(np.abs(padded_img)+1)
    centered_log = np.zeros((p,q))
    # 중심점 이동
    for x in range(p):
        for y in range(q):
            centered_log[x, y] = dft2d_log[x,y]*((-1)**(x+y))

    ''' freq. domain에서 정의됨 '''
    def GLPF(img, cutoff, low, high):
        m,n = img.shape
        f= np.zeros((m,n))

        U0 = int(m/2)
        V0 = int(n/2)

        D0 = cutoff

        for u in range(m):
            for v in range(n):
                d_uv = np.sqrt((u-U0)**2+(v-V0)**2)
                l, h = low, high
                f[u, v] = (h-l)*(1-np.exp(-d_uv**2/(2*D0**2)))+l

        return f

    # DFT
    dft2d = np.fft.fft2(centered_log)

    # 필터와 이미지를 곱함
    homorophic_filter = GLPF(centered_log, cutoff=30, low=0.7, high=1.2)
    filtered_log_dft = np.multiply(homorophic_filter, dft2d)

    # IDFT
    filtered_log = np.fft.ifft2(filtered_log_dft)
    filtered = np.zeros(filtered_log.shape, dtype=np.float32)

    # 다시 중심 이동
    for x in range(p):
        for y in range(q):
            filtered[x,y] = ((-1)**(x+y))*np.real(filtered_log[x,y])

    # exponential
    result = np.exp(filtered) -1
    result_new = result[:m, :n]

    normalized_result = (result_new - np.min(result_new)) / (np.max(result_new) - np.min(result_new)) * 255
    normalized_result = normalized_result.astype(np.uint8)

    return normalized_result