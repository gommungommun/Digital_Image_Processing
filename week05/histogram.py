import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('bean.jpg', 0)

# 히스토그램 평활화
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()

cdf_norm = np.ma.masked_equal(cdf, 0)
cdf_norm = (cdf_norm-cdf_norm.min())*255/(cdf_norm.max()-cdf_norm.min())
cdf_final = np.ma.filled(cdf_norm, 0).astype(np.uint8)

img_eq = cdf_final[img]

fig = plt.figure()



ax0 = fig.add_subplot(211)
ax0.set_title('original') 
ax0.hist(img.ravel(), 256, [0, 256], color='b')

ax1 = fig.add_subplot(212)
ax1.set_title('histogram_eq') 
ax1.hist(img_eq.ravel(), 256, [0, 256], color='r')

plt.tight_layout()  # 서브플롯 간 간격 자동 조정
plt.show()

cv2.imshow('original', img)
cv2.imshow('histogram_eq', img_eq)
cv2.waitKey()
cv2.destroyAllWindows()