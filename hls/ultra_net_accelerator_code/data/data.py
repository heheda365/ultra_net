import cv2
import numpy as np

img = cv2.imread("boat6_0.jpg")
# img = cv2.resize(img, (320, 160))

print(img.shape)

img.tofile("boat6_0.bin")
