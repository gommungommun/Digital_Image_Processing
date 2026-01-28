import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('gatsby.jpg', 0)
m,n = img.shape
p,q = m*2, n*2

# 1. zero-padding
padded_img = np.zeros((p,q))
padded_img[:m, :n] = img

# 2. log를 취함
dft2d_log = np.log(np.abs(padded_img)+1)
centered_log = np.zeros((p,q))
# 3. 중심점 이동
for x in range(p):
    for y in range(q):
        centered_log[x, y] = dft2d_log[x,y]*((-1)**(x+y))

''' freq. domain에서 정의됨 '''
def GLPF(img, cutoff, parameter):
    m,n = img.shape
    f= np.zeros((m,n))

    U0 = int(m/2)
    V0 = int(n/2)

    D0 = cutoff

    for u in range(m):
        for v in range(n):
            d_uv = np.sqrt((u-U0)**2+(v-V0)**2)
            l, h = parameter
            f[u, v] = (h-l)*(1-np.exp(-d_uv**2/(2*D0**2)))+l

    return f

# 4. DFT
dft2d = np.fft.fft2(centered_log)

# 5. 필터와 이미지를 곱함
homorophic = GLPF(centered_log, cutoff=100, parameter=[0.75, 2.0])
filtered_log_dft = dft2d*homorophic

# 6. IDFT
filtered_log = np.fft.ifft2(filtered_log_dft)
filtered = np.zeros(filtered_log.shape, dtype=np.float32)

# 7. 다시 중심 이동
for x in range(m):
    for y in range(n):
        filtered[x,y] = ((-1)**(x+y))*np.real(filtered_log[x,y])

# 8. exponential
result = np.exp(filtered) -1
result_new = result[:m, :n]

''' image normalization 중요! '''
# 0~255 사이로 정규화
normalized_result = (result_new - np.min(result_new)) / (np.max(result_new) - np.min(result_new)) * 255
normalized_result = normalized_result.astype(np.uint8)

# 이미지 표시
plt.imshow(normalized_result, cmap='gray'), plt.axis('off')
plt.title('homomorphic filtering')
plt.show()