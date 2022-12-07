import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./harris4.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (700, 800))
ori_img = np.copy(img)
# th, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# th, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 149, 255, cv2.THRESH_BINARY)
# print(th)

rect = (20, 20, img.shape[1]-50, img.shape[0]-50)
labels = np.zeros(img.shape[:2], np.uint8)
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((labels == 2) | (labels == 0), 0, 1).astype('uint8')
mask_img = img*mask2[:, :, np.newaxis]

# mask = np.zeros(img.shape[:2], np.uint8)
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# rect = (20, 20, img.shape[1]-20, img.shape[0]-20)
# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
# mask_img = img*mask2[:, :, np.newaxis]

kernel = np.ones((5, 5), np.uint8)
mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)


corners = cv2.goodFeaturesToTrack(mask2, 4, 0.05, 10)
cor = []
for c in corners:
    x, y = c[0]
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    cor.append([x, y])

cor.sort(key=lambda x : x[1])
up = [cor[0], cor[1]]
down = [cor[2], cor[3]]
up.sort(key=lambda x : x[0])
down.sort(key=lambda x : x[0])
corners = np.array([[up[0]], [up[1]], [down[0]], [down[1]]])

width = 600
height = 800
dst_pts = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)
m = cv2.getPerspectiveTransform(corners, dst_pts)
warp_img = cv2.warpPerspective(ori_img, m, (width, height))

cv2.imshow('original', ori_img)
cv2.imshow('warp', warp_img)
cv2.imshow('binary', mask_img)
cv2.imshow('corner', img)
cv2.imshow('mask', mask2 * 255)
cv2.imwrite('warp_img.jpg', warp_img)
cv2.waitKey()
cv2.destroyAllWindows()

