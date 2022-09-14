import cv2

img = cv2.imread('../images/lenna.png', 1)
cv2.imshow('Test', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
