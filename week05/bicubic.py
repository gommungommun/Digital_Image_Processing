import cv2
import numpy as np

img = cv2.imread('orig_img.jpg')
if img is None:
    print("이미지를 불러올 수 없습니다.")
    exit()

h, w, c = img.shape
scale = 3
result = np.zeros((h*scale, w*scale, c))

for channel in range(c):
    for y in range(h*scale):
        for x in range(w*scale):
            # 원본 좌표의 계산
            xo = x/scale
            yo = y/scale
            x0 = int(xo)-1
            x1 = int(xo)
            x2 = int(xo)+1
            x3 = int(xo)+2
            y0 = int(yo)-1
            y1 = int(yo)
            y2 = int(yo)+1
            y3 = int(yo)+2
            
            # 범위 확인
            x0 = max(0, min(x0, w-1))
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w-1))
            x3 = max(0, min(x3, w-1))
            y0 = max(0, min(y0, h-1))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h-1))
            y3 = max(0, min(y3, h-1))
            
            # x_index = [x0, x1, x2, x3]
            # y_index = [y0, y1, y2, y3]
            
            tx = xo - int(xo)
            ty = yo - int(yo)
            tx3 = tx**3
            tx2 = tx**2
            ty3 = ty**3
            ty2 = ty**2
            
            x_cubic = [
                -0.5 * tx3 + tx2 - 0.5 * tx,
                1.5 * tx3 - 2.5 * tx2 + 1,
                -1.5 * tx3 + 2 * tx2 + 0.5 * tx,
                0.5 * tx3 - 0.5 * tx2
            ]
            
            y_cubic = [
                -0.5 * ty3 + ty2 - 0.5 * ty,
                1.5 * ty3 - 2.5 * ty2 + 1,
                -1.5 * ty3 + 2 * ty2 + 0.5 * ty,
                0.5 * ty3 - 0.5 * ty2
            ]
            
            temp = []
            for i in range(4):
                y_pixel = [y0, y1, y2, y3][i]
                row_pixels = [
                    img[y_pixel, x0, channel],
                    img[y_pixel, x1, channel],
                    img[y_pixel, x2, channel],
                    img[y_pixel, x3, channel]
                ]
                interpolation = sum(x_cubic[j] * row_pixels[j] for j in range(4))
                temp.append(interpolation)
            
            final_result = sum(y_cubic[i]*temp[i] for i in range(4))
            final_result = np.clip(final_result, 0, 255)
            result[y, x, channel] = np.round(final_result)

result = result.astype(np.uint8)
cv2.imshow('Original', img)
cv2.imshow('result', result)
cv2.waitKey()
cv2.destroyAllWindows()