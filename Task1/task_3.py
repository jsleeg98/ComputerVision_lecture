import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./TestImage/image_House256rgb.png', 0).astype(np.float32) / 255

radius = int(input('필터 반지름 입력 : '))
filter_type = input('high or low pass : ')

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])

if filter_type == 'high':
    mask = np.zeros(fft.shape, np.uint8)
    mask = cv2.circle(mask, (128, 128), radius, (255, 255), -1)
elif filter_type == 'low':
    mask = np.ones(fft.shape, np.uint8)
    mask *= 255
    mask = cv2.circle(mask, (128, 128), radius, (0, 0), -1)
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])

filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
mask_new = np.dstack((mask, np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)))


plt.figure(figsize=(8, 4))
plt.subplot(131)
plt.axis('off')
plt.title('Original')
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.axis('off')
plt.title('Mask')
plt.imshow(mask_new, cmap='gray')

plt.subplot(133)
plt.axis('off')
plt.title('Filtered')
plt.imshow(filtered, cmap='gray')

plt.tight_layout()
plt.show()