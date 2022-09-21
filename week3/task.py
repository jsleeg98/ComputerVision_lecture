import cv2
import numpy as np

# task 1
img = cv2.imread('../images/lena.png')
cv2.imshow('lena', img)
cv2.waitKey()
cv2.destroyAllWindows()

# task 2-1
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grey_eq = cv2.equalizeHist(grey)
cv2.imshow('grey', grey)
cv2.imshow('equalized grey', grey_eq)
cv2.waitKey()
cv2.destroyAllWindows()

# task 2-2
grey_float32 = grey.astype(np.float32) / 255
gamma = 0.5
corrected_image = np.power(grey_float32, gamma)
cv2.imshow('grey', grey_float32)
cv2.imshow('corrected_image', corrected_image)
cv2.waitKey()
cv2.destroyAllWindows()

# task 3
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]
median_filter = cv2.medianBlur(h, 7)
gaussian_filter = cv2.GaussianBlur(s, (9, 9), 0)
bilat = cv2.bilateralFilter(v, -1, 0.1, 10)
cv2.imshow('h', h)
cv2.imshow('s', s)
cv2.imshow('v', v)
cv2.imshow('h_median_filter', median_filter)
cv2.imshow('s_gaussian_filter', gaussian_filter)
cv2.imshow('v_bilateral_filter', bilat)
cv2.waitKey()
cv2.destroyAllWindows()