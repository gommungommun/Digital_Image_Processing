import numpy as np
import cv2
import matplotlib.pyplot as plt

def morphology(img, method, kernel, k_size, inverse=False):
    h, w =img.shape
    pad = k_size // 2
    if not inverse:
        pad_img =  np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    else:
        pad_img =  np.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
        
        
    result = img.copy()
    
    for i in range(h):
        for j in range(w):
            if method == 1 :
                result[i,j] = erosion(pad_img[i:i+k_size, j:j+k_size], kernel)
            elif method == 2:
                result[i,j] = dilation(pad_img[i:i+k_size, j:j+k_size], kernel)
                
    return result
                
def erosion(boundary, kernel):
    out = np.where((kernel>0) & (boundary>0), 1, 0)
    if (out==kernel).all():
        return 255
    else:
        return 0
    
def dilation(boundary, kernel):
    boundary = boundary * kernel
    if np.max(boundary) != 0:
        return 255
    else:
        return 0