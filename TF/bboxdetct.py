#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:17:39 2019

@author: ran
"""
#bboxdetc to detect the bounding box of cell in imwited image(oriimg) by using Difference of Gaussian

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
#import cv2
#import matplotlib.pyplot as plt
import numpy as np
from combinedbbox import compare2bbox
#from PIL import Image,ImageDraw

def bboxdetc(image):
    #image_path    = "%s%s%s"%(image_path_head,image_seq,image_path_tail)
    #def FirstFrameBBox(ima):#input first frame file_name with ''
    
    
    #image=cv2.imread(oriimg)
    
    
    #image=cv2.imread('/home/ran/Trails/PhC-C2DH-U373/02T/img/0000.jpg') #first frame jpg path
    
    
    
    #img=cv2.imread(oriimg)
    
    #image = cv2.imread('/home/ran/Pictures/t0001.jpg')
    #image=cv2.imread(ima)
    image_gray = rgb2gray(image)
    image_gray = np.power(image_gray/float(np.max(image_gray)), 1/3) # Gamma correction with parameter 1/3
    blobs_dog = blob_dog(image_gray, max_sigma=60, threshold=.15) #default 10 .02
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    #print(blobs_dog)
    blobs_dog=np.array(blobs_dog)
    blobs_dog.astype(int)
    
    FFBB=np.zeros((len(blobs_dog),4))
    FFBB[0][1]=1

    #print('Image size is:',image.shape[1],'*',image.shape[0])
    for p in range(len(blobs_dog)):
        #blobs_dog=[y,x,r] transfer to bbox as: x,y,shiftx, shifty
        r=int(blobs_dog[p][2])
        FFBB[p][0]=blobs_dog[p][1]-r
        FFBB[p][1]=blobs_dog[p][0]-r
        FFBB[p,2]=2*r
        FFBB[p,3]=2*r
        #if is out of image size
        if FFBB[p][0]+FFBB[p,2]>image.shape[1]:
            FFBB[p,2]=image.shape[1]-FFBB[p][0]
        if FFBB[p][1]+FFBB[p,3]>image.shape[0]:
            FFBB[p,3]=image.shape[0]-FFBB[p][1]
    
    FFBB=compare2bbox(FFBB)
    #BBox Size Filtering
    dindex=[]
    
    for p in range(0,len(FFBB)):
            
            if FFBB[p][2]*FFBB[p][3]>20000 or FFBB[p][2]<13:#default p2<20
                flag=1
                dindex.append(p)
    FFBB=np.delete(FFBB,dindex,0) 
    
    
    #fop=open('/home/ran/Trails/PhC-C2DH-U373/02T/firstframe1.txt',mode='w')
    #fop=open('/home/ran/Data/exp_F0001 1-113/firstframe1.txt',mode='w')     
    #FFBB=np.array(FFBB)
    #zer0=str('0,0,0,0')
    #print('First Frame Bounding Box is: [x,y,shiftx,shifty]')
   
    return(FFBB)