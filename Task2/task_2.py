import cv2
import numpy as np

image_set = input('image_set : ')

if image_set == 'boat':
    img0 = cv2.imread('./stitching/boat1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./stitching/boat2.jpg', cv2.IMREAD_COLOR)
    img0 = cv2.resize(img0, (0, 0), fx=0.3, fy=0.3)
    img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)

    sift = cv2.xfeatures2d.SIFT_create(50)
    kps0, fea0 = sift.detectAndCompute(img0, None)
    kps1, fea1 = sift.detectAndCompute(img1, None)

    show_img_sift0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_sift1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('sift0', show_img_sift0)
    cv2.imshow('sift1', show_img_sift1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    surf = cv2.xfeatures2d.SURF_create(10000)
    surf.setExtended(True)
    surf.setNOctaves(3)
    surf.setNOctaveLayers(10)
    surf.setUpright(False)

    kps0, fea0 = surf.detectAndCompute(img0, None)
    kps1, fea1 = surf.detectAndCompute(img1, None)

    show_img_surf0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_surf1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('surf0', show_img_surf0)
    cv2.imshow('surf1', show_img_surf1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    orb = cv2.ORB_create()
    orb.setMaxFeatures(200)

    kps0, fea0 = orb.detectAndCompute(img0, None)
    kps1, fea1 = orb.detectAndCompute(img1, None)

    show_img_orb0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_orb1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('orb0', show_img_orb0)
    cv2.imshow('orb1', show_img_orb1)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif image_set == 'budapest':
    img0 = cv2.imread('./stitching/budapest1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./stitching/budapest2.jpg', cv2.IMREAD_COLOR)
    # img0 = cv2.resize(img0, (0, 0), fx=0.3, fy=0.3)
    # img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)

    sift = cv2.xfeatures2d.SIFT_create(50)
    kps0, fea0 = sift.detectAndCompute(img0, None)
    kps1, fea1 = sift.detectAndCompute(img1, None)

    show_img_sift0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_sift1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('sift0', show_img_sift0)
    cv2.imshow('sift1', show_img_sift1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    surf = cv2.xfeatures2d.SURF_create(10000)
    surf.setExtended(True)
    surf.setNOctaves(3)
    surf.setNOctaveLayers(10)
    surf.setUpright(False)

    kps0, fea0 = surf.detectAndCompute(img0, None)
    kps1, fea1 = surf.detectAndCompute(img1, None)

    show_img_surf0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_surf1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('surf0', show_img_surf0)
    cv2.imshow('surf1', show_img_surf1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    orb = cv2.ORB_create()
    orb.setMaxFeatures(200)

    kps0, fea0 = orb.detectAndCompute(img0, None)
    kps1, fea1 = orb.detectAndCompute(img1, None)

    show_img_orb0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_orb1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('orb0', show_img_orb0)
    cv2.imshow('orb1', show_img_orb1)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif image_set == 'newspaper':
    img0 = cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./stitching/newspaper2.jpg', cv2.IMREAD_COLOR)
    # img0 = cv2.resize(img0, (0, 0), fx=0.3, fy=0.3)
    # img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)

    sift = cv2.xfeatures2d.SIFT_create(50)
    kps0, fea0 = sift.detectAndCompute(img0, None)
    kps1, fea1 = sift.detectAndCompute(img1, None)

    show_img_sift0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_sift1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('sift0', show_img_sift0)
    cv2.imshow('sift1', show_img_sift1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    surf = cv2.xfeatures2d.SURF_create(10000)
    surf.setExtended(True)
    surf.setNOctaves(3)
    surf.setNOctaveLayers(10)
    surf.setUpright(False)

    kps0, fea0 = surf.detectAndCompute(img0, None)
    kps1, fea1 = surf.detectAndCompute(img1, None)

    show_img_surf0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_surf1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('surf0', show_img_surf0)
    cv2.imshow('surf1', show_img_surf1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    orb = cv2.ORB_create()
    orb.setMaxFeatures(200)

    kps0, fea0 = orb.detectAndCompute(img0, None)
    kps1, fea1 = orb.detectAndCompute(img1, None)

    show_img_orb0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_orb1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('orb0', show_img_orb0)
    cv2.imshow('orb1', show_img_orb1)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif image_set == 's':
    img0 = cv2.imread('./stitching/s1.jpg', cv2.IMREAD_COLOR)
    img1 = cv2.imread('./stitching/s2.jpg', cv2.IMREAD_COLOR)
    # img0 = cv2.resize(img0, (0, 0), fx=0.3, fy=0.3)
    # img1 = cv2.resize(img1, (0, 0), fx=0.3, fy=0.3)

    sift = cv2.xfeatures2d.SIFT_create(50)
    kps0, fea0 = sift.detectAndCompute(img0, None)
    kps1, fea1 = sift.detectAndCompute(img1, None)

    show_img_sift0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_sift1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('sift0', show_img_sift0)
    cv2.imshow('sift1', show_img_sift1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    surf = cv2.xfeatures2d.SURF_create(10000)
    surf.setExtended(True)
    surf.setNOctaves(3)
    surf.setNOctaveLayers(10)
    surf.setUpright(False)

    kps0, fea0 = surf.detectAndCompute(img0, None)
    kps1, fea1 = surf.detectAndCompute(img1, None)

    show_img_surf0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_surf1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('surf0', show_img_surf0)
    cv2.imshow('surf1', show_img_surf1)
    cv2.waitKey()
    cv2.destroyAllWindows()

    orb = cv2.ORB_create()
    orb.setMaxFeatures(200)

    kps0, fea0 = orb.detectAndCompute(img0, None)
    kps1, fea1 = orb.detectAndCompute(img1, None)

    show_img_orb0 = cv2.drawKeypoints(img0, kps0, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img_orb1 = cv2.drawKeypoints(img1, kps1, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('orb0', show_img_orb0)
    cv2.imshow('orb1', show_img_orb1)
    cv2.waitKey()
    cv2.destroyAllWindows()

# matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, False)
# matches = matcher.match(fea0, fea1)

# pts0 = np.float32([kps0[m.queryIdx].pt for m in matches]).reshape(-1, 2)
# pts1 = np.float32([kps1[m.trainIdx].pt for m in matches]).reshape(-1, 2)
# H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 3.0)
#
# print(mask)

