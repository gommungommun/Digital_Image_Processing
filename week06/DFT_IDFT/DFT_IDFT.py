import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('img32.jpg', 0)
plt.imshow(img, cmap='gray'), plt.axis('off')
plt.title('original')
plt.show()
m,n = img.shape
p,q = m*2, n*2

padded_img = np.zeros((p,q))
padded_img[:m, :n] = img
padded_img_new = np.zeros((p,q))

for x in range(p):
    for y in range(q):
        padded_img_new[x, y] = padded_img[x, y]*((-1)**(x+y))

def DFT(padded_img):
    M,N = padded_img.shape
    H = np.zeros((M,N), dtype=complex)

    for u in range(M):
        for v in range(N):
            tot = 0.0
            for x in range (M):
                for y in range(N):
                    e = np.exp(-2j*np.pi*((u*x)/M+(v*y)/N))
                    tot += e*padded_img[x,y]
            H[u,v] = tot
    return H

def IDFT(dft_img):
    M,N = dft_img.shape
    H = np.zeros((M,N), dtype=complex)
    
    for x in range(M):
        for y in range(N):
            tot = 0.0
            for u in range(M):
                for v in range(N):
                    e = np.exp(2j*np.pi*((u*x)/M+(v*y)/N))
                    tot += e*dft_img[u, v]
            H[x,y] = tot
    
    return H

dft2d = DFT(padded_img_new)
plt.imshow(dft2d.real, cmap='gray'), plt.axis('off')
plt.title('DFT')
plt.show()

idft2d = IDFT(dft2d)

for x in range(p):
    for y in range(q):
        idft2d[x, y] = idft2d[x, y]*((-1)**(x+y))

plt.imshow(idft2d[:int(p/2),:int(q/2)].real, cmap='gray'), plt.axis('off')
plt.title('idft')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')

plt.subplot(122)
plt.imshow(idft2d[:m,:n].real, cmap='gray')
plt.axis('off')
plt.title('IDFT Result')

plt.tight_layout()
plt.show()