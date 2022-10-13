import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../images/lena.png').astype(np.float32) / 255
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# data = image.reshape((-1, 3))

image_xy = np.ones((image.shape[0], image.shape[1]))
tmp_c = np.arange(1, image.shape[0]+1).astype(np.float32)
tmp_r = tmp_c.reshape(image.shape[0], 1).astype(np.float32)
image_c = np.multiply(image_xy, tmp_c)
image_r = np.multiply(image_xy, tmp_r)

RGBXY_data = np.dstack((image, image_r, image_c))
A, B, C = RGBXY_data.shape
RGBXY_data = RGBXY_data.reshape((-1, 5)).astype(np.float32)


RGB_data = image_lab.reshape((-1, 3))

num_classes = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
_, labels, centers = cv2.kmeans(RGB_data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)

num_classes = 8
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
_, labels, centers = cv2.kmeans(RGBXY_data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

RGBXY_lab = centers[labels.flatten()].reshape((A,B,C))
RGBXY_lab = RGBXY_lab[:,:,:3]
RGBXY_image = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2BGR)


image_r = np.arange(image.shape[0])
image_c = np.arange(image.shape[1])
image_r = image_r.reshape(image.shape[0], 1)
image_c = image_c.reshape(image.shape[1], 1)


plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image[:, :, [2, 1, 0]])
plt.subplot(132)
plt.axis('off')
plt.title('RGB')
plt.imshow(segmented[:, :, [2, 1, 0]])
plt.subplot(133)
plt.axis('off')
plt.title('RGBXY')
plt.imshow(RGBXY_image[:, :, ::-1])

plt.tight_layout
plt.show()