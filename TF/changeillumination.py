#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:11:19 2019

@author: ran
"""

import  cv2
import numpy as np
img=cv2.imread('/home/ran/Trails/N2DH-GOWH1/01/img/0045.jpg')
#img=cv2.imread('/home/ran/Pictures/t0006.jpg')
#cv2.imshow('original_img',img)
'''



rows,cols,channels=img.shape
dst=img.copy()

a=1.5
b=100
for i in range(rows):
    for j in range(cols):
        for c in range(3):
            color=img[i,j][c]*a+b
            if color>255:           # 防止像素值越界（0~255）
                dst[i,j][c]=255
            elif color<0:           # 防止像素值越界（0~255）
                dst[i,j][c]=0
                
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#!/usr/bin/python
# coding:utf-8




#img = cv2.imread('gamma0.jpg',0)

img1 = np.power(img/float(np.max(img)), 1/2)
cv2.imwrite('/home/ran/Pictures/t10007.jpg',img1)

#cv2.imshow('gamma=1/1.8',img1)
cv2.waitKey(0)
