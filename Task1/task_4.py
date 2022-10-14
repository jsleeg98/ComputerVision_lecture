import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./TestImage/image_Peppers512rgb.png', 0)

thres, image_otsu = cv2.threshold(image, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

morph = int(input('1.Erosion 2.Dilation, 3.Opening, 4.Closing : '))
count = int(input('횟수 : '))

if morph == 1:
    result = cv2.morphologyEx(image_otsu, cv2.MORPH_ERODE, (3, 3), iterations=count)
elif morph == 2:
    result = cv2.morphologyEx(image_otsu, cv2.MORPH_DILATE, (3, 3), iterations=count)
elif morph == 3:
    result = cv2.morphologyEx(image_otsu, cv2.MORPH_OPEN, (3, 3), iterations=count)
elif morph == 4:
    result = cv2.morphologyEx(image_otsu, cv2.MORPH_CLOSE, (3, 3), iterations=count)

plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.axis('off')
plt.title('Original')
plt.imshow(image_otsu, cmap='gray')

plt.subplot(122)
plt.axis('off')
plt.title('Result')
plt.imshow(result, cmap='gray')

plt.tight_layout()
plt.show()