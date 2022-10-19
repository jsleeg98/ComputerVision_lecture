import cv2
import numpy as np

img = cv2.imread('../images/scenetext01.jpg', cv2.IMREAD_COLOR)
corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

corners = cv2.dilate(corners, None)

show_img = np.copy(img)
show_img[corners>0.1*corners.max()] = [0, 0, 255]

corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
harris_show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

# cv2.imshow('Harris corner detector', show_img)
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

fast = cv2.FastFeatureDetector_create(30, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
kp = fast.detect(img)

show_img = np.copy(img)
for p in cv2.KeyPoint_convert(kp):
    cv2.circle(harris_show_img, tuple(p), 4, (0, 255, 0), cv2.FILLED)


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(img_gray, 100, 0.05, 10)

for c in corners:
    x, y = c[0]
    cv2.circle(harris_show_img, (x, y), 4, (255, 0, 0), -1)

detector = cv2.xfeatures2d.SIFT_create(50)
keypoints, descriptors = detector.detectAndCompute(img, None)

for p in keypoints:
    harris_show_img = cv2.circle(harris_show_img, (int(p.pt[0]), int(p.pt[1])), 4, (0, 255, 255), -1)

cv2.imshow('Result', harris_show_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()


