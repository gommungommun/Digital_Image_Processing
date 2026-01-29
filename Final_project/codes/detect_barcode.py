import cv2
import numpy as np

def morphology(img, method, kernel):
    k_height, k_width = kernel.shape
    h, w = img.shape
    
    # 패딩 크기를 높이와 너비에 맞게 각각 계산
    pad_h, pad_w = k_height // 2, k_width // 2
    
    # 패딩 적용 (Dilation과 Erosion 모두 0으로 패딩)
    pad_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # 결과 이미지를 원본과 같은 크기로 초기화
    result = img.copy()

    for i in range(h):
        for j in range(w):
            # 윈도우(boundary)를 커널 크기에 맞게 추출
            window = pad_img[i:i + k_height, j:j + k_width]
            if method == 1: # Erosion
                result[i, j] = ersion(window, kernel)
            elif method == 2: # Dilation
                result[i, j] = dilation(window, kernel)
            
    return result

def closing(img, kernel):
    # k_size 인자 제거
    dilation_img = morphology(img, 2, kernel)
    closing_img = morphology(dilation_img, 1, kernel)
    return closing_img

# opening 함수는 이 코드에서 사용되지 않지만, 일관성을 위해 수정
def opening(img, kernel):
    # k_size 인자 제거
    erosion_img = morphology(img, 1, kernel)
    opening_img = morphology(erosion_img, 2, kernel)
    return opening_img

def ersion(boundary, kernel):
    out  = np.where((kernel>0)&(boundary>0), 1, 0)
    if(out==kernel).all():
        return 255
    else:
        return 0
    
def dilation(boundary, kernel):
    boundary = np.multiply(boundary, kernel)
    if np.max(boundary)!=0:
        return 255
    else: 
        return 0
def otsu_implement(img):
    """
    NumPy만을 이용하여 오츠 알고리즘을 직접 구현합니다.
    (요청하신 변수명으로 모두 수정한 최종 버전)
    """
    # 히스토그램 계산
    hist, _ = np.histogram(img.ravel(), bins=256, range=[0, 256])
    tot_pixel = img.size
    tot_intensity = np.dot(np.arange(256), hist)
    background_pixel = 0
    background_intensity = 0
    opt_threshold = 0
    max_var = 0.0
    
    # 각 픽셀의 임계값마다 0~255 값 사이
    for t in range(256):
        background_pixel += hist[t]
        background_intensity += t * hist[t]
        
        # 해당 픽셀 자리에서 개수가 0이면
        if background_pixel == 0:
            continue
        
        # 전체 픽셀에서 배경 픽셀을 빼면 오브젝트 픽셀 -> 0에서 시작해서 255까지 남은 픽셀의 개수
        obj_pixel = tot_pixel - background_pixel
        
        if obj_pixel == 0:
            break
        
        # 각각의 빈도
        w_b = background_pixel / tot_pixel
        w_f = obj_pixel / tot_pixel
        
        # 남아있는 픽셀이 존재할 경우 
        # 평균은 현재 존재하는 픽셀의 평균값(현재 픽셀 값) / 해당 픽셀 값
        if background_pixel > 0:
            mean_b = background_intensity / background_pixel
        else:
            mean_b = 0

        if obj_pixel > 0:
            # obj의 intensity를 가지고 평균을 냄
            mean_f = (tot_intensity - background_intensity) / obj_pixel
        else:
            mean_f = 0

        temp_var = w_b * w_f * ((mean_b - mean_f) ** 2)
        
        # 새롭게 지정된 분산이 이전 최대값보다 크면 갱신됨
        if temp_var > max_var:
            max_var = temp_var
            opt_threshold = t
            
    return opt_threshold

def Detect_barcode(gray_img):
    # 그래디언트 계산 
    gray_img = np.uint8(gray_img)
    gradX = cv2.Scharr(gray_img, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Scharr(gray_img, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # 스무딩, binary로 변경함
    blurred = cv2.blur(gradient, (9, 9))
    optimal_threshold = otsu_implement(blurred)
    _, thresh = cv2.threshold(blurred, optimal_threshold, 255, cv2.THRESH_BINARY)
    
    kernel_close = np.ones((7, 21), dtype=np.uint8)
    closed = closing(thresh, kernel_close)
    
    # 작은 노이즈 제거를 위한 3x3 커널
    kernel_noise = np.ones((3, 3), dtype=np.uint8)

    # noise제거를 강력하게 해야 하므로 erosion과 dilation을 각각  4변 연속으로 수행
    eroded_result = closed.copy()
    for _ in range(4):
        eroded_result = morphology(eroded_result, 1, kernel_noise) # method=1: Erosion
    dilated_result = eroded_result.copy()
    for _ in range(4):
        dilated_result = morphology(dilated_result, 2, kernel_noise) # method=2: Dilation
    
    final_morph_img = dilated_result
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(final_morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    if len(contours) > 0:
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # 엣지 박스 표시
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

    return output_img
