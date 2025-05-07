#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:11:19 2019

@author: ran
"""
from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import numpy as np

#img=cv2.imread('/home/ran/Trails/N2DH-GOWH1/01/img/0000.jpg')


image_path_head   = "/home/ran/Trails/N2DH-GOWH1/01/img/"
#image_path_head   = "/home/ran/Trails/PhC-C2DH-U373/ideo/img/"
#image_path_head   = "/home/ran/Trails/PhC-C2DH-U373/02T/img/"
#image_path_head   = "/home/ran/Data/exp_F0001 1-113/data/data/"
image_path_tail   = ".jpg"
image_path_extra  = "'"

image_write_head = "/home/ran/Trails/N2DH-GOWH1/01/adjustedimg/"

for idx in range(0,91):
    idx2 = idx+1
    idxname = "%04d" % idx
    image_seq     = idxname
    image_idx     = str(idxname)
    image_path    = "%s%s%s"%(image_path_head,image_seq,image_path_tail)
    image_write_path = "%s%s%s"%(image_write_head,image_seq,image_path_tail)
    idx1 = idx
    exec('image2=cv2.imread("/home/ran/Trails/N2DH-GOWH1/01/img/'+image_idx+'.jpg")')
    #cv2.imshow('test',img1)
    img2 = rgb2gray(image2)
    image2= np.power(img2/float(np.max(img2)), 1/3)*float(np.max(img)) # Gamma correction with parameter 1/3
    #cv2.imshow('result',image1)
    cv2.imwrite(image_write_path,image2)

#cv2.imshow('gamma=1/1.8',img1)
#cv2.waitKey(0)
