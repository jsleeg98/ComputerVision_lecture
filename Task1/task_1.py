import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./TestImage/image_Peppers512rgb.png', cv2.IMREAD_COLOR)

BGR = input('R, G, B 중 하나의 채널을 입력해주세요 : ').upper()

b, g, r = cv2.split(img)

if BGR == 'B':
    img_2 = b
    plt.figure(figsize=(16, 8))

    plt.subplot(231)
    plt.axis('off')
    plt.title(f'Original')
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.subplot(232)
    plt.axis('off')
    plt.title(f'{BGR} Original')
    plt.imshow(img_2)

    plt.subplot(233)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Original Histogram')
    plt.hist(img_2.ravel(), 256, [0, 256])

    plt.subplot(235)
    img_2_eq = cv2.equalizeHist(img_2)
    plt.axis('off')
    plt.title(f'{BGR} Equalized')
    plt.imshow(img_2_eq)

    plt.subplot(236)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Equalized Histogram')
    plt.hist(img_2_eq.ravel(), 256, [0, 256])

    plt.subplot(234)
    plt.title('Final')
    plt.axis('off')
    img[:, :, 0] = img_2_eq
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.tight_layout()
    plt.show()

elif BGR == 'G':
    img_2 = g
    plt.figure(figsize=(16, 8))

    plt.subplot(231)
    plt.axis('off')
    plt.title(f'Original')
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.subplot(232)
    plt.axis('off')
    plt.title(f'{BGR} Original')
    plt.imshow(img_2)

    plt.subplot(233)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Original Histogram')
    plt.hist(img_2.ravel(), 256, [0, 256])

    plt.subplot(235)
    img_2_eq = cv2.equalizeHist(img_2)
    plt.axis('off')
    plt.title(f'{BGR} Equalized')
    plt.imshow(img_2_eq)

    plt.subplot(236)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Equalized Histogram')
    plt.hist(img_2_eq.ravel(), 256, [0, 256])

    plt.subplot(234)
    plt.title('Final')
    plt.axis('off')
    img[:, :, 1] = img_2_eq
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.tight_layout()
    plt.show()

elif BGR == 'R':
    img_2 = r
    plt.figure(figsize=(16, 8))

    plt.subplot(231)
    plt.axis('off')
    plt.title(f'Original')
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.subplot(232)
    plt.axis('off')
    plt.title(f'{BGR} Original')
    plt.imshow(img_2)

    plt.subplot(233)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Original Histogram')
    plt.hist(img_2.ravel(), 256, [0, 256])

    plt.subplot(235)
    img_2_eq = cv2.equalizeHist(img_2)
    plt.axis('off')
    plt.title(f'{BGR} Equalized')
    plt.imshow(img_2_eq)

    plt.subplot(236)
    plt.xlabel('pixel value')
    plt.title(f'{BGR} Equalized Histogram')
    plt.hist(img_2_eq.ravel(), 256, [0, 256])

    plt.subplot(234)
    plt.title('Final')
    plt.axis('off')
    img[:, :, 2] = img_2_eq
    plt.imshow(img[:, :, [2, 1, 0]])

    plt.tight_layout()
    plt.show()
