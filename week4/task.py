import math

import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

img = cv2.imread('../images/lena.png')

# task 1
KSIZE = 11
ALPHA = 2
kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
print(kernel.shape, kernel.dtype, kernel.sum())

filtered = cv2.filter2D(img, -1, kernel)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(img[:, :, [2, 1, 0]])
plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout(True)
plt.show()

cv2.imshow('before', img)
cv2.imshow('after', filtered)
cv2.waitKey()
cv2.destroyAllWindows()

# filter image 처리
filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

# task 2
dx = cv2.Sobel(filtered_gray, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(filtered_gray, cv2.CV_32F, 0, 1)
img_sobel = cv2.addWeighted(dx, 1, dy, 1, 0)
print(f'img_sobel type : {img_sobel.dtype}')

plt.figure(figsize=(8, 3))
plt.subplot(141)
plt.axis('off')
plt.title('image')
plt.imshow(filtered_gray, cmap='gray')
plt.subplot(142)
plt.axis('off')
plt.title(r'$\frac{dI}{dx}$')
plt.imshow(dx, cmap='gray')
plt.subplot(143)
plt.axis('off')
plt.title(r'$\frac{dI}{dy}$')
plt.imshow(dy, cmap='gray')
plt.subplot(144)
plt.axis('off')
plt.title('sobel')
plt.imshow(img_sobel, cmap='gray')
plt.tight_layout(True)
plt.show()

# task 3
kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())

img_gabor = cv2.filter2D(filtered_gray, -1, kernel)
img_gabor = img_gabor.astype(np.float32)
print(f'img_gabor type : {img_gabor.dtype}')

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(filtered_gray, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('kernel')
plt.imshow(kernel, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(img_gabor, cmap='gray')
plt.tight_layout(True)
plt.show()

# task 4
def onChange(pos):
    global img_diff
    _, img_thres = cv2.threshold(img_diff, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('Track', img_thres)

img_diff = cv2.absdiff(img_sobel, img_gabor)
cv2.namedWindow('Track')
cv2.createTrackbar('threshold', 'Track', 0, 255, onChange)
cv2.imshow('Track', img_diff)
cv2.waitKey()
cv2.destroyAllWindows()

# task 5
opened = cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 5)
closed = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 5)
cv2.imshow('opened', opened)
cv2.imshow('closed', closed)
cv2.waitKey()
cv2.destroyAllWindows()

# task 6
img = cv2.imread('../images/lena.png', 0).astype(np.float32) / 255

fft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
sz = 25
mask = np.zeros(fft.shape, np.uint8)
mask[img.shape[0]//2-sz:img.shape[0]//2+sz,
    img.shape[1]//2-sz:img.shape[1]//2+sz, :] = 1
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)))

plt.figure()
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(img, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filtered, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('mask')
plt.imshow(mask_new*255, cmap='gray')
plt.tight_layout(True)
plt.show()