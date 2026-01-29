import cv2
import numpy as np
import matplotlib.pyplot as plt
from morphology_1 import morphology

def getting_kernel():
    k = np.ones((3, 3), dtype=np.int8)
    k[1, 1] = -1
    return k

def getting_image():
    img = np.zeros((7,15))
    img[1, 1:5] = 1
    img[1, 10:14] = 1
    img[2, 1:14] = 1
    img[3, 1:5] = 1
    img[3, 6:9] = 1
    img[3, 10:14] = 1
    img[4, 1:14] = 1
    img[5, 1:5] = 1
    img[5, 10:14] = 1

    img = img*255

    return img.astype(np.uint8)

def hitmiss(img, kernel):
    img_c = np.where(img > 0, 0, 255).astype(np.uint8)
    B1 = np.where(kernel > 0, 1, 0).astype(np.uint8)
    B2 = np.where(kernel < 0, 1, 0).astype(np.uint8)

    A1 = morphology(img, 1, B1, 3)
    A2 = morphology(img_c, 1, B2, 3, inverse=True)
    out = np.where((A1 > 0) & (A2 > 0), 255, 0).astype(np.uint8)
    return out

if __name__ == "__main__":
    org_img = getting_image()
    kernel = getting_kernel()
    X = org_img.copy()
    res = hitmiss(X, kernel)
    

    plt.subplot(131); plt.imshow(org_img, cmap='gray'); plt.title('Original image'); plt.axis('off')
    plt.subplot(132); plt.imshow(kernel, cmap='gray'); plt.title('kernel'); plt.axis('off')
    plt.subplot(133); plt.imshow(res, cmap='gray'); plt.title('result'); plt.axis('off')

    plt.show() 