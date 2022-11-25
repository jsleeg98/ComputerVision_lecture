import cv2
import numpy as np

imgs = ['./stitching/boat1.jpg',
        './stitching/budapest1.jpg',
        './stitching/newspaper1.jpg',
        './stitching/s1.jpg']
for file in imgs:
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    corners = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)

    corners = cv2.dilate(corners, None)

    show_img = np.copy(img)
    show_img[corners>0.1*corners.max()] = [0, 0, 255]

    corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    harris_show_img = np.hstack((show_img, cv2.cvtColor(corners, cv2.COLOR_GRAY2BGR)))

    show_img = cv2.resize(show_img, (0, 0), fx=0.5, fy=0.5)

    canny = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)
    canny = cv2.resize(canny, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow('Canny edge detector', canny)
    cv2.imshow('Harris corner detector', show_img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()