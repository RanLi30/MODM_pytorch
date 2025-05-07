#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:09:31 2020

@author: ran
"""

import os
import cv2
import numpy as np
from math import sqrt
from skimage import data




image_path   = '/home/ran/Trails/PhC-C2DH-U373/CODE/01/01jpg/0000.jpg'

gt_path_head = '/home/ran/Trails/PhC-C2DH-U373/RESULTS/TE_01_GT/V5/'
gt_file_tail = ".txt"
gt_path_tail = ".tif" 
gt_path_extra = "'"
CellNum = 6
redius=1
image=cv2.imread(image_path)
size=image.shape 


##### Initializtion ####
example=np.loadtxt(gt_path_head+'1'+gt_file_tail)
length=len(example)
bg = np.zeros([size[0], size[1]], np.uint16)
for i in range(0,len(example)):
    bg=np.array(bg, dtype=np.uint16)
    gtname= "man_track"+"%03d" % i
    gt_path="%s%s%s"%(gt_path_head,gtname,gt_path_tail)
    cv2.imwrite(gt_path, bg)

################################
for cidx in range(1,CellNum+1):
  
    
    GTFile=str(cidx)
    src = "%s%s%s"%(gt_path_head,GTFile,gt_file_tail)
    
    #src = os.path.join(config.otb_data_dir,seq_name,'groundtruth_rect1.txt')
    lines=np.loadtxt(src)
    #lines = cell index / frame (start from 1) / x-axis / y-axis


    for rect in lines:
        frame=rect[1]-1
        gtname= "man_track"+"%03d" % frame
        gt_path="%s%s%s"%(gt_path_head,gtname,gt_path_tail)        
        image1=cv2.imread(gt_path,-1)
        #gt_rects.append(rect)
        center=(int(rect[2]),int(rect[3]))        
        image1=cv2.circle(image1, center,redius,(rect[0],rect[0],rect[0]), -1)
        image2 = np.array(image1, dtype=np.uint16)
        cv2.imwrite(gt_path, image2)


