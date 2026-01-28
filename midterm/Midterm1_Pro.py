"""
Image processing project exam 1
---------------------------------
Name:
Student ID:
---------------------------------
Problem 1 Gamma Correction and Contrast Enhancement
"""
def getGammaCorrection(img):
    gamma = 3.0
    
    img_normalized = img.astype(np.float32) / 255.0
    gamma_corrected = np.power(img_normalized, 1.0 / gamma)
    res = (gamma_corrected * 255.0).astype(np.uint8)
    
    return res

# 1-2) Contrast Enhancement (수식 직접 구현)
def ContrasteEnhancement(img):
    # intensity: 기울기, bias: y절편
    intensity, bias = 1.64, -114.8

    res = img.astype(np.float32) * intensity + bias
    
    res = np.clip(res, 0, 255).astype(np.uint8)
    
    return res 

if __name__=='__main__':
    # 이미지 로드 (Grayscale)
    image = cv2.imread('puppy.jpg', 0)


    gamma_img = getGammaCorrection(image)
    result = ContrasteEnhancement(gamma_img)

    # 1-3) 보정 전/후 Histogram 출력
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Histogram')
    plt.hist(image.ravel(), 256, [0, 256], color='gray')

    plt.subplot(1, 2, 2)
    plt.title('Result Histogram')
    plt.hist(result.ravel(), 256, [0, 256], color='blue')

    plt.show()

    cv2.imshow('Original', image)
    cv2.imshow('Result', result)
    cv2.imwrite('Answer1.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()