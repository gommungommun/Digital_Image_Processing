import cv2
import matplotlib.pyplot as plt
import Homorophic as Ho
import DFT
import median
import img_histogram as hist
import detect_barcode
import crop_area
import contrast_enhance 

barcode1 = cv2.imread('barcode1.jpg', cv2.COLOR_RGB2BGR)
barcode2 = cv2.imread('barcode2.jpg', 0)
barcode3 = cv2.imread('barcode3.jpg', 0)

# 바코드가 있는 영역만 크롭
barcode1 = crop_area.crop_barcode_box(barcode1)
# filtering을 적용하기 위해 흑백 이미지로 바꿈
barcode1 = cv2.cvtColor(barcode1, cv2.COLOR_RGB2GRAY)


# # barcode들에 각각 필터 적용
# result1 = Ho.homorophic(barcode1)
# result2 = median.median_blur(barcode2, 7)
result3 = DFT.notch_filter(barcode3)

# # sharpening 결과 시각화
# # 결과 1: Homomorphic Filterusalgus
# cv2.imshow('Result 1 - Homomorphic', result1)

# # 결과 2: Median Blur
# cv2.imshow('Original 2', barcode2)
# cv2.imshow('Result 2 - Median', result2)

# 결과 3: Notch filter
plt.imshow(result3, cmap='gray')
plt.title('Result 3 - Notch filter')
plt.axis('off')
plt.show()

# # filtering 후 histogram 
# hist.hist_img(result1)
# hist.hist_img(result2)
# hist.hist_img(result3)

# # # hist eq
# result1 = cv2.equalizeHist(result1)
# cv2.imshow('Result 1 - Homorophic', result1)

# # get binary
result3 = contrast_enhance.get_binary(result3, 110)
cv2.imshow('Result 3', result3)

# final1 = detect_barcode.Detect_barcode(barcode1)
# final2 = detect_barcode.Detect_barcode(result2)
final3 = detect_barcode.Detect_barcode(result3)

# cv2.imshow('detect barcode1', final1)
# cv2.imshow('detect barcode2', final2)
plt.imshow(final3, cmap='gray')
plt.title('detect barcode3')
plt.axis('off')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()