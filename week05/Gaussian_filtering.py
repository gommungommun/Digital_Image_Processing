import cv2
import matplotlib as plt
import numpy as np

img = cv2.imread('orig_img.jpg', 0)
h, w = img.shape

zero_padding = np.zeros((h+2, w+2))
zero_padding[1:1+h, 1:1+w] = img

gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]])/16

result = np.zeros((h, w))

for i in range(h):
    for j in range(w):
        window = zero_padding[i:i+3, j:j+3]
        result[i,j] = np.sum(window*gaussian_filter)


cv2.imshow('result', result.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()