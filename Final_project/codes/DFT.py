import cv2
import numpy as np
import matplotlib.pyplot as plt

def notch_filter(image, radius=10):
    # freq_mag 함수를 통해 노이즈가 있을 것으로 생각되는 포인트들을 구함
    points = [
        (320, 105), (283, 140), (265, 157),
        (265, 195), (283, 208), (320, 245),
        (175, 105), (210, 140), (230, 157),
        (175, 245), (210, 210), (230, 195),
        (320, 100), (180, 100), (320, 300), 
        (180, 300), (108, 100), (380, 100),
        (108, 250), (380, 250), (60, 250),
        (60, 100), (430, 100), (430, 250),
        (180, 20), (320, 20), (180, 340),
        (320, 340), (490, 250), (490, 100),
        (10, 100), (10, 250), (470, 150),
        (470, 210), (30, 210), (30, 150),
        (290, 270), (290, 100), (220, 100),
        (220, 270)
    ]
    # 이미지를 float 타입으로 변환 후 푸리에 변환(FFT) 수행
    img_float32 = np.float32(image)
    # 원래 아래의 
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 노치 필터 커널 생성
    rows, cols = image.shape
    crow, ccol = rows // 2 , cols // 2

    # 실수부와 허수부를 위한 2채널 마스크 생성 (기본값: 1로 채움)
    mask = np.ones((rows, cols, 2), np.float32)

    # 주어진 각 점과 그에 대한 대칭점에 노치(원)를 생성
    for point in points:
        x, y = point
        # 대칭점 계산 (FFT 스펙트럼의 중심 대칭)
        sym_x = 2 * ccol - x
        sym_y = 2 * crow - y
        
        # 주어진 점과 대칭점에 반지름(radius) 크기의 원(값=0)을 그려 노치를 생성
        cv2.circle(mask, (x, y), radius, 0, -1)
        cv2.circle(mask, (sym_x, sym_y), radius, 0, -1)

    # 3. 필터 적용
    fshift = np.multiply(dft_shift, mask)
    
    # 4. 역 푸리에 변환(Inverse FFT)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 5. 이미지 정규화
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    filtered_image = np.uint8(img_back)
    
    return filtered_image