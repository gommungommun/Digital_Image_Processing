import numpy as np
import cv2
import matplotlib.pyplot as plt

def median_blur(img, kernel_size=3):
    H, W = img.shape
    new_img = np.zeros_like(img)
    offset = kernel_size // 2

    for y in range(H):
        for x in range(W):
            neighbors = []
            for ky in range(-offset, offset + 1):
                for kx in range(-offset, offset + 1):
                    ny, nx = y + ky, x + kx
                    ny = max(0, min(H - 1, ny))
                    nx = max(0, min(W - 1, nx))
                    neighbors.append(img[ny, nx])
            
            neighbors.sort()
            new_img[y, x] = neighbors[len(neighbors) // 2]
    
    return new_img