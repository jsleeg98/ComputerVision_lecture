import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../images/lena.png', help='Image path.')
parser.add_argument('--out_png', default='../images/lena_task.png', help='Output image path')
params = parser.parse_args()
image = cv2.imread(params.path)
image_to_show = np.copy(image)
original_image = image

mouse_pressed = False
s_x = s_y = e_x = e_y = -1


def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y

def rect_mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, image

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.rectangle(image_to_show, (s_x, s_y),(x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y
        image = image_to_show


def line_mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, image

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.line(image_to_show, (s_x, s_y),(x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y
        image = image_to_show

def arrowedline_mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, image

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_to_show = np.copy(image)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(image)
            cv2.arrowedLine(image_to_show, (s_x, s_y),(x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y
        image = image_to_show

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('image', image_to_show)
    k = cv2.waitKey(1)

    if k == ord('r'):
        cv2.setMouseCallback('image', rect_mouse_callback)

    elif k == ord('l'):
        cv2.setMouseCallback('image', line_mouse_callback)

    elif k == ord('a'):
        cv2.setMouseCallback('image', arrowedline_mouse_callback)

    elif k == ord('w'):
        cv2.imwrite(params.out_png, image_to_show, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    elif k == ord('c'):
        image = original_image
        image_to_show = original_image

    elif k == 27:
        break

cv2.destroyAllWindows()

