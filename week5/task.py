import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 1
image = cv2.imread('../images/lena.png', 0)

otsu_thr, otsu_mask = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Estimated threshold (Otsu): ', otsu_thr)

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('Otsu threshold')
plt.imshow(otsu_mask, cmap='gray')
plt.tight_layout()
plt.show()


# 2
dst, contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(otsu_mask, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('external')
plt.imshow(image_external, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('internal')
plt.imshow(image_internal, cmap='gray')
plt.tight_layout()
plt.show()



# connectivity = 8
# num_labels, labelmap = cv2.connectedComponents(otsu_mask, connectivity, cv2.CV_32S)
#
# otsu_mask_2 = np.hstack((otsu_mask, labelmap.astype(np.float32)/(num_labels-1)))
# cv2.imshow('Connected components', otsu_mask_2)
# cv2.waitKey()
# cv2.destroyAllWindows()
# 3
image_3 = cv2.imread('task_image.png', 0)
# cv2.imshow('test', image_3)
# cv2.waitKey()
# cv2.destroyAllWindows()

otsu_thr, otsu_mask_2 = cv2.threshold(image_3, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
otsu_mask_2 = 255 * otsu_mask_2
otsu_mask_inv = 255 - otsu_mask_2


# cv2.imshow('test', otsu_mask_inv)
# cv2.waitKey()
# cv2.destroyAllWindows()

connectivity = 100
output = cv2.connectedComponentsWithStats(otsu_mask_inv, connectivity, cv2.CV_32S)

num_labels, labelmap, stats, centers = output
li_labels = []


colored = np.full((otsu_mask_inv.shape[0], otsu_mask_inv.shape[1], 3), 0, np.uint8)


while True:
    img = cv2.cvtColor(otsu_mask_inv, cv2.COLOR_GRAY2BGR)
    cv2.imshow('Connected components', np.hstack((img, colored)))

    k = cv2.waitKey(1)

    if k == 32:
        colored = np.full((otsu_mask_inv.shape[0], otsu_mask_inv.shape[1], 3), 0, np.uint8)
        li_randint = []
        for i in range(5):
            a = random.randint(1, num_labels)
            while a in li_randint:
                a = random.randint(1, num_labels)
            li_randint.append(a)

        print(li_randint)

        for l in li_randint:
            colored[labelmap==l] = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))


    elif k == 27:
        break

cv2.destroyAllWindows()

# 5
distmap = cv2.distanceTransform(otsu_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

plt.figure()
plt.imshow(distmap, cmap='gray')
plt.show()

