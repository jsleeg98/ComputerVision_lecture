import cv2

images = []
images.append(cv2.imread('./stitching/newspaper1.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/newspaper2.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/newspaper3.jpg', cv2.IMREAD_COLOR))
images.append(cv2.imread('./stitching/newspaper4.jpg', cv2.IMREAD_COLOR))

stitcher = cv2.createStitcher()
ret, pano = stitcher.stitch(images)

if ret == cv2.STITCHER_OK:
    pano = cv2.resize(pano, dsize=(0, 0), fx=0.5, fy=0.5)
    cv2.imshow('panorama', pano)
    cv2.waitKey()

    cv2.destroyAllWindows()
else:
    print('Error during stiching')