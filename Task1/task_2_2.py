import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

def on_diameter_change(pos):
    global bilat, diameter, SigmaColor, SigmaSpace
    if pos == 0:
        diameter = -1
    else:
        diameter = pos
    print(f'diameter : {diameter}, SigmaColor : {SigmaColor}, SigmaSpace : {SigmaSpace} ')
    bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)
    cv2.imshow('Result', bilat)

def on_SigmaColor_change(pos):
    global bilat, diameter, SigmaColor, SigmaSpace
    SigmaColor = pos
    print(f'diameter : {diameter}, SigmaColor : {SigmaColor}, SigmaSpace : {SigmaSpace} ')
    bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)
    cv2.imshow('Result', bilat)

def on_SigmaSpace_change(pos):
    global bilat, diameter, SigmaColor, SigmaSpace
    SigmaSpace = pos
    print(f'diameter : {diameter}, SigmaColor : {SigmaColor}, SigmaSpace : {SigmaSpace} ')
    bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)
    cv2.imshow('Result', bilat)

image = cv2.imread('./TestImage/Lena.png').astype(np.float32) / 255
noised = (image + 0.2 * np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0, 1)
diameter = -1
SigmaColor = 0
SigmaSpace = 0
bilat = cv2.bilateralFilter(noised, diameter, SigmaColor, SigmaSpace)

cv2.namedWindow('Result')
cv2.createTrackbar('diameter', 'Result', 0, 10, on_diameter_change)
cv2.createTrackbar('SigmaColor', 'Result', 0, 300, on_SigmaColor_change)
cv2.createTrackbar('SigmaSpace', 'Result', 0, 20, on_SigmaSpace_change)
cv2.imshow('Result', bilat)
cv2.waitKey()
cv2.destroyAllWindows()



