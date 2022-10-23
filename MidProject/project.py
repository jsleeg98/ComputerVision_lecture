import cv2
import numpy as np
from PIL import Image, ImageEnhance

img = cv2.imread('test_image_5.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (600, 800))
show_img = np.copy(img)  # 영상 출력 이미지

mouse_pressed = False
y = x = w = h = 0

# 객체가 있는 사각형 그리는 callback 함수
def mouse_callback(event, _x, _y, flags, param):
    global show_img, x, y, w, h, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        x, y = _x, _y
        show_img = np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            show_img = np.copy(img)
            cv2.rectangle(show_img, (x, y), (_x, _y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        # 어느 방향으로든 사각형을 그릴 수 있도록 처리
        if _x < x:
            _x, x = x, _x
        if _y < y:
            _y, y = y, _y
        w, h = _x - x, _y - y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)

    if k == ord('a') and not mouse_pressed:  # 사각형을 그린 경우
        if w * h > 0:
            break

cv2.destroyAllWindows()

# grabCut
labels = np.zeros(img.shape[:2], np.uint8)
labels, bgdModel, fgdModel = cv2.grabCut(img, labels, (x, y, w, h), None, None, 5, cv2.GC_INIT_WITH_RECT)

show_img = np.copy(img)
show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 3  # 배경 어두움 처리

cv2.imshow('image', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

label = cv2.GC_BGD
lbl_clrs = {cv2.GC_BGD: (0, 0, 0), cv2.GC_FGD: (255, 255, 255)}

# 배경, 물체 세부 조정 callback 함수
def mouse_callback(event, x, y, flags, param):
    global mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
        cv2.circle(show_img, (x, y), 5, lbl_clrs[label], cv2.FILLED)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(labels, (x, y), 5, label, cv2.FILLED)
            cv2.circle(show_img, (x, y), 5, lbl_clrs[label], cv2.FILLED)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)

    if k == ord('a') and not mouse_pressed:  # a를 누른 경우, 객체, 배경 조정 내용 출력, 이후 미세 조정 가능
        labels, bgdModel, fgdModel = cv2.grabCut(img, labels, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        show_img = np.copy(img)
        show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] //= 4  # 배경 부분 어둡게 처리
    elif k == ord('1'):
        label = cv2.GC_FGD - label
    elif k == ord('n'):  # 미세 조정 완료 후 다음 단계
        break

cv2.destroyAllWindows()

mask_bg = np.where((labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD), 1, 0).astype('uint8')  # 배경 마스크

def on_sigma_change(pos):  # trackbar를 이용하여 GaussianBlur sigma 조절
    global img, result, mask_bg, show_img
    sigma = pos
    if sigma == 0:  # sigma가 0인 경우 1로 바꾸어 오류 제거
        sigma = 1
    print(f'sigma : {sigma}')
    img_blur = cv2.GaussianBlur(img, (0, 0), sigma)
    bg_blur = img_blur * mask_bg[:, :, np.newaxis]
    result = bg_blur + show_img
    cv2.imshow('result', result)

obj = show_img[(labels == cv2.GC_PR_BGD) | (labels == cv2.GC_BGD)] = 0  # 배경 부분 0으로 처리
img_blur = cv2.GaussianBlur(img, (0, 0), 1)
bg_blur = img_blur * mask_bg[:, :, np.newaxis]
result = bg_blur + show_img
cv2.namedWindow('result')
cv2.createTrackbar('sigma', 'result', 1, 20, on_sigma_change)

while True:
    cv2.imshow('result', result)
    k = cv2.waitKey(1)
    if k == ord('s'):  # 현재 이미지 저장
        cv2.imwrite('result.jpg', result)
    elif k == ord('n'):  # 종료
        break
cv2.destroyAllWindows()

# v = 5
# # sharpening_mask = np.array([[0, -1, 0], [-1, v, -1], [0, -1, 0]])
# # img_sharpen = cv2.filter2D(result, -1, sharpening_mask)
# result = Image.open('result.jpg')
# enhancer = ImageEnhance.Sharpness(result)
# img_sharpen = enhancer.enhance(v)
#
# def on_value_change(pos):  # trackbar를 이용하여 GaussianBlur sigma 조절
#     global sharpening_mask, result, img_sharpen
#     value = pos
#     # sharpening_mask = np.array([[0, -1, 0], [-1, v, -1], [0, -1, 0]])
#     # img_sharpen = cv2.filter2D(result, -1, sharpening_mask)
#     img_sharpen = enhancer.enhance(v)
#     cv2.imshow('sharpen', img_sharpen)
#
# cv2.namedWindow('sharpen')
# cv2.createTrackbar('value', 'sharpen', 1, 100, on_value_change)
#
# while True:
#     cv2.imshow('sharpen', img_sharpen)
#     k = cv2.waitKey(1)
#     if k == ord('n'):  # 종료
#         break
# cv2.destroyAllWindows()








