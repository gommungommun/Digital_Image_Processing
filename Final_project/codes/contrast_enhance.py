import cv2
import numpy as np

def get_binary(img, T):
    H,W = img.shape
    result = np.zeros((H,W))
    result[img >= T] = 255
    return result