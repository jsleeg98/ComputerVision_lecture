import argparse

import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='../images/lenna.png', help='Image path.')
params = parser.parse_args()


img = cv2.imread(params.path)

#Check if image was successfullly read.
assert img is not None

print(f'read {params.path}')
print(f'shape: {img.shape}')
print(f'dtype: {img.dtype}')

img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
assert img is not None
print(f'read {params.path} as grayscale')
print(f'shape: {img.shape}')
print(f'dtype: {img.dtype}')

cv2.imshow('Test', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
