import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('moon.png', 0)
m,n = img.shape
p,q = m*2, n*2
padded_img = np.zeros((p,q))
padded_img_new = np.zeros((p,q))

padded_img[:m, :n] = img

# Laplacian 함수 수정
def Laplacian(img):
    m,n = img.shape
    H = np.zeros((m, n))
    
    for u in range(m):
        for v in range(n):
            u2 = np.power((u-m/2),2)
            v2 = np.power((v-n/2),2)
            H[u,v] = -(u2+v2)
    return H

for x in range(p):
    for y in range(q):
        padded_img_new[x, y] = padded_img[x, y] * ((-1)**(x+y))

dft2d = np.fft.fft2(padded_img_new)
dft2d_ = np.log(np.abs(dft2d))

laplacian = Laplacian(dft2d)
plt.imshow(laplacian, cmap='gray'), plt.axis('off')
plt.title('laplacian')
plt.show()

G = np.multiply(laplacian, dft2d)
dft2d_ = np.log(np.abs(G))
plt.imshow(dft2d_.real, cmap='gray'), plt.axis('off')
plt.title('laplacian spectrum')
plt.show()

idft2d = np.fft.ifft2(G)

for x in range(p):
    for y in range(q):
        idft2d[x,y] = idft2d[x,y]*((-1)**(x+y))

idft2d = idft2d[:m,:n]
min,max = np.min(idft2d), np.max(idft2d)
idft2d = (idft2d-min)/(max-min)*255.0

sharpening = np.add(img, idft2d.real)
smoothing = np.subtract(img, idft2d.real)

plt.imshow(idft2d.astype(np.uint8), cmap='gray'),plt.axis('off')
plt.title('laplacian')
plt.show()

plt.imshow(sharpening, cmap='gray'), plt.axis('off')
plt.title('sharpening')
plt.show()

plt.imshow(smoothing, cmap='gray'), plt.axis('off')
plt.title('smoothing')
plt.show()