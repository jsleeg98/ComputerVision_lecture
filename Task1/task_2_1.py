import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./TestImage/Lena.png').astype(np.float32) / 255
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)

diameter = int(input('diameter 값 입력 : '))
SigmaColor = float(input('SigmaColor 값 입력 : '))
SigmaSpace = float(input('SigmaSpace 값 입력 : '))

bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.axis('off')
plt.title('Noised')
plt.imshow(noised[:, :, [2, 1, 0]])

plt.subplot(122)
plt.axis('off')
plt.title('Bilateral')
plt.imshow(bilat[:, :, [2, 1, 0]])

plt.tight_layout()
plt.show()