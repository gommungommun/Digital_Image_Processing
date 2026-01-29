import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_barcode_box(img):
    # 2. 색상 공간 변환 (BGR -> HSV)
    # 흰색과 같은 무채색을 검출하기 위해 BGR보다 HSV 색상 공간이 더 효과적입니다.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. 흰색 영역을 위한 임계값 설정
    # H(색상), S(채도), V(명도)
    # 흰색은 채도가 낮고, 명도가 높습니다.
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])

    # 4. 마스크 생성
    # 설정한 흰색 범위에 해당하는 픽셀만 흰색(255)으로, 나머지는 검은색(0)으로 만듭니다.
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 5. 가장 큰 컨투어(輪郭) 찾기
    # 마스크에서 경계선을 찾아 가장 큰 영역을 바코드의 흰색 배경으로 간주합니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("흰색 영역을 찾지 못했습니다.")
        return None

    # 면적이 가장 큰 컨투어를 찾습니다.
    main_contour = max(contours, key=cv2.contourArea)

    # 6. 컨투어를 감싸는 사각형 좌표 얻기 및 크롭
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # 원본 이미지에서 해당 좌표를 이용해 잘라냅니다.
    cropped_img = img[y:y+h, x:x+w]

    return cropped_img

def crop_rining_artifact(img):
    M,N = img.shape
    P,Q = M-60, N-60

    result = img[30:M-30, 30:N-30]

    return result