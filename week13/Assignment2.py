import cv2
import numpy as np
import matplotlib.pyplot as plt
from morphology_1 import morphology

# Thinning용 커널들 (-1: false, 0: don’t care, 1: true)
B1 = np.array([[-1,-1,-1],
               [ 0, 1, 0],
               [ 1, 1, 1]], dtype=int)
B2 = np.array([[ 0,-1,-1],
               [ 1, 1,-1],
               [ 1, 1, 0]], dtype=int)
B3 = np.rot90(B1, 3)
B5 = np.rot90(B3, 3)
B7 = np.rot90(B5, 3)
B4 = np.rot90(B2, 3)
B6 = np.rot90(B4, 3)
B8 = np.rot90(B6, 3)

thin_kernel_list = [B1, B2, B3, B4, B5, B6, B7, B8]

def getimg_thinning():
    img = np.zeros((5, 11))
    img[0,0] = 1
    img[0,8:11] = 1
    img[1,0:2] = 1
    img[1,7:9] = 1
    img[2,1:8] = 1
    img[3,0:2] = 1
    img[3,7] = 1
    img[4,0] = 1
    img[4,7:9] = 1
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
    I = getimg_thinning()

    thick_kernel_list = thin_kernel_list.copy()
    thick_kernel_list.reverse()

    results = [None] * 9

    results[0] = I.copy()

    X = I.copy()
    for i in range(8):
        X_c = np.where(X > 0, 0, 255).astype(np.uint8)
        for kernel in thick_kernel_list:
            hitmiss_c = hitmiss(X_c, kernel)
            hitmiss_c_c = np.where(hitmiss_c > 0, 0, 255).astype(np.uint8)
            X_c = np.where((X_c > 0) & (hitmiss_c_c > 0), 255, 0).astype(np.uint8)
            
        X = np.where(X_c > 0, 0, 255).astype(np.uint8)
        results[i + 1] = X.copy()

    fig, ress = plt.subplots(1,  9, figsize=(14, 3))
    for idx in range(9):
        res = ress[idx]
        res.imshow(results[idx], cmap="gray")
        if idx == 0:
            res.set_title("Original")
        else:
            res.set_title(f"Iter {idx}")
    plt.tight_layout()
    plt.show()