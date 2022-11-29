import cv2 
import matplotlib.pyplot as plt

img1 = cv2.imread('./stitching/dog_a.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./stitching/dog_b.jpg', cv2.IMREAD_GRAYSCALE)

corners1 = cv2.goodFeaturesToTrack(img1, 100, 0.05, 10)
corners2 = cv2.goodFeaturesToTrack(img2, 100, 0.05, 10)

pts, status, errors = cv2.calcOpticalFlowPyrLK(
    img1, img2, corners1, None, winSize=(15, 15), maxLevel=5,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



good_pts = pts[status == 1]
tracks = good_pts

img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for p in tracks:
    cv2.circle(img2, (p[0], p[1]), 3, (0, 255, 0), -1)

cv2.imshow('frame', img2)
cv2.waitKey()
cv2.destroyAllWindows()

