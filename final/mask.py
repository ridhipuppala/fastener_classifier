from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2
from math import floor, ceil, pi
import re
from time import time
from time import sleep
from PIL import Image


img = cv2.imread('./cropimgs_png/7.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,5)
gray = cv2.dilate(gray, None, iterations=10)
gray = cv2.erode(gray, None, iterations=10)


#ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(th2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

ims, cnts, hierarchys = cv2.findContours(th3.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))
c = max(cnts, key=cv2.contourArea)
cv2.drawContours(img, cnts, -1, (0,255,0), 5)
epsilon = 0.1*cv2.arcLength(c,True)
approx = cv2.approxPolyDP(c,epsilon,True)
hull = cv2.convexHull(c)
x,y,w,h = cv2.boundingRect(c)
print(x,y,w,h)
cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)

# M = cv2.moments(c)
# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
# ((x, y), radius) = cv2.minEnclosingCircle(c)
# cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
# cv2.circle(img, center, 3, (0, 0, 255), -1)

cv2.namedWindow('Test Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Test Image' , 500,500)
cv2.imshow('Test Image',img)
#cv2.imshow('Test Image',img)

k = cv2.waitKey(0) & 0xFF
if k == 27:
	pass
	cv2.destroyAllWindows()
