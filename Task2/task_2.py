import cv2
import numpy as np

img0 = cv2.imread('./stitching/newspaper2.jpg', cv2.IMREAD_COLOR)
img1 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
hL, wL = img0.shape[:2]
hR, wR = img1.shape[:2]

# surf = cv2.xfeatures2d.SURF_create(10000)
# surf.setExtended(True)
# surf.setNOctaves(3)
# surf.setNOctaveLayers(10)
# surf.setUpright(False)
#
# keyPoints0, descriptors0 = surf.detectAndCompute(img0, None)
# keyPoints1, descriptors1 = surf.detectAndCompute(img1, None)
#
# show_img0 = cv2.drawKeypoints(img0, keyPoints0, None, (255, 0, 0),
#                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img1 = cv2.drawKeypoints(img1, keyPoints0, None, (255, 0, 0),
#                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



sift = cv2.xfeatures2d.SIFT_create()
kp0, descriptors0 = sift.detectAndCompute(img0, None)
kp1, descriptors1 = sift.detectAndCompute(img1, None)

show_img0_sift = cv2.drawKeypoints(img0, kp0, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
show_img1_sift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

matcher = cv2.DescriptorMatcher_create('BruteForce')
matches = matcher.knnma(descriptors1, descriptors0)

# sorted_matches = sorted(matches, key = lambda x : x.distance)
# res = cv2.drawMatches(img0, kp0, img1, kp1, sorted_matches[:30], None, flags=2)

good_matches = []
for m in matches:
    if len(m) == 2 and m[0].distance > m[1].distance * 0.75:
        good_matches.append((m[0].trainIdx, m[0].queryIdx))

print(good_matches)

if len(good_matches) > 4:
    pts0 = np.float32([kp0[i].pt for (i, _) in good_matches])
    pts1 = np.float32([kp1[i].pt for (i, _) in good_matches])
    matrix, status = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
    panorama = cv2.warpPerspective(img1, matrix, (wR + wL, hR))
    panorama[0:hL, 0:wL] = img0

cv2.imshow('panorama_sift', panorama)
cv2.waitKey()
cv2.destroyAllWindows()

# orb = cv2.ORB_create()
# orb.setMaxFeatures(200)
#
# kp0_orb = orb.detect(img0, None)
# kp0_orb, descriptors0_orb = orb.compute(img0, kp0_orb)
#
# kp1_orb = orb.detect(img1, None)
# kp1_orb, descriptors1_orb = orb.compute(img1, kp1_orb)
#
# show_img0_orb = cv2.drawKeypoints(img0, kp0_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# show_img1_orb = cv2.drawKeypoints(img1, kp1_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imshow('img0 SURF descriptors', show_img0)
# cv2.imshow('img1 SURF descriptors', show_img1)
# cv2.imshow('img0 SIFT descriptors', show_img0_sift)
# cv2.imshow('img1 SIFT descriptors', show_img1_sift)
# cv2.imshow('img0 orb descrioptors', show_img0_orb)
# cv2.imshow('img1 orb descrioptors', show_img1_orb)
# cv2.waitKey()
# cv2.destroyAllWindows()