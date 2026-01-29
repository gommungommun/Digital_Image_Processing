import cv2
import matplotlib.pyplot as plt
import numpy as np

def hist_img(img):
    plt.hist(img.ravel(), 256,[0,256])
    plt.show()