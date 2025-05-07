#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:38:30 2018

@author: ran
"""

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
from bboxdetct import bboxdetc
from combinedbbox import compare2bbox
#image_path    = "%s%s%s"%(image_path_head,image_seq,image_path_tail)
#def FirstFrameBBox(ima):#input first frame file_name with ''
#image=cv2.imread('/home/ran/Trails/PhC-C2DH-U373/CODE/02/02jpg/0000.jpg') #first frame jpg path
#image=cv2.imread('/home/ran/Trails/C2DL-MSC/01/img/0000.jpg')
#image=cv2.imread('/home/ran/Data/exp_F0001 1-113/data/0001.jpg')

#image = cv2.imread('/home/ran/Pictures/t0001.jpg')
#image=cv2.imread(ima)
def DoG(image):    
    image_gray=image
    #image_gray = rgb2gray(image)
    image_gray = np.power(image_gray/float(np.max(image_gray)), 1/3) # Gamma correction with parameter 1/3
    blobs_dog = blob_dog(image_gray, max_sigma=20, threshold=.12) #U373-01 60 .12
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    #print(blobs_dog)
    blobs_dog=np.array(blobs_dog)
    blobs_dog.astype(int)
    
    FFBB=np.zeros((len(blobs_dog),4))
    FFBB[0][1]=1
    ImageNumber=115
    print('Image size is:',image.shape[1],'*',image.shape[0])
    for p in range(len(blobs_dog)):
        #blobs_dog=[y,x,r] transfer to bbox as: x,y,shiftx, shifty
        r=int(blobs_dog[p][3])
        FFBB[p][0]=blobs_dog[p][1]-2*r
        FFBB[p][1]=blobs_dog[p][0]-2*r
        FFBB[p,2]=4*r
        FFBB[p,3]=4*r
        #if is out of image size
        if FFBB[p][0]+FFBB[p,2]>image.shape[1]:
            FFBB[p,2]=image.shape[1]-FFBB[p][0]
        if FFBB[p][1]+FFBB[p,3]>image.shape[0]:
            FFBB[p,3]=image.shape[0]-FFBB[p][1]
    
    #FFBB=compare2bbox(FFBB)
    #BBox Size Filtering
    dindex=[]
    
    for p in range(0,len(FFBB)):
            
            if FFBB[p][3]>100:
                flag=1
                dindex.append(p)
    FFBB=np.delete(FFBB,dindex,0) 
    return FFBB

def drawfirstbbox(FFBB):
    fop=open('/home/ran/Trails/HL60/firstframe1.txt',mode='w')
    #fop=open('/home/ran/Trails/BF-C2DL-MuSC/Test/01/firstframe1.txt',mode='w')
    #fop=open('/home/ran/Data/exp_F0001 1-113/firstframe1.txt',mode='w')     
    #FFBB=np.array(FFBB)
    zer0=str('0,0,0,0')
    print('First Frame Bounding Box is: [x,y,shiftx,shifty]')
    
    
    #for eachline in FFBB:
    for idx in range(0,len(FFBB)): 
            #去掉文本行里面的空格、\t、数字（其他有要去除的也可以放到' \t1234567890'里面）
            a = FFBB[idx]
            idx1=idx+1
            '''
            a=str(a) 
            #lines = filter(lambda ch: ch not in ' \t\n1234567890,', eachline) 
            b=a.replace('[','')
            c=b.replace(']','')
            d=c.replace('.','')
            d.lstrip()
            print (c)
            '''
            a=a.astype(np.int)
            FFBBoutput=",".join(str(i) for i in a)#output split with ',' no , at the ends
            print(FFBBoutput)            
            #file_save.write(np.array2string(bbox))
            fop.write(FFBBoutput)
            fop.write('\n')
            
             
            '''
            gt1=open('/home/ran/Trails/PhC-C2DH-U373/02T/groundtruth_rect1.txt',mode='w')
            gt1.write(FFBBoutput)
            gt1.write('\n')
            for iman in range(1,114):
                gt1.write(zer0)
                gt1.write('\n')
            '''
            
            idx1=str(idx1)
            image_seq     = idx1
            
            exec("gt"+idx1+"=open('/home/ran/Trails/A-10-run05/CODE/01/groundtruth_rect"+idx1+".txt',mode='w')")
            #exec("gt"+idx1+"=open('/home/ran/Trails/N2DH-GOWH1/02/groundtruth_rect"+idx1+".txt',mode='w')")
            exec('gt'+idx1+'.write(FFBBoutput)')
            exec("gt"+idx1+".write('\\n')")
            
            for iman in range(1,115):
                exec('gt'+idx1+'.write(zer0)')
                exec('gt'+idx1+'.write("\\n")')
            
    
            exec('gt'+idx1+'.close')
    #FFBB1 =  d.split()
    #print(FFBB1)
    #FFBB=list(int(d))
             
    
    fop.close
    #print the image
    #print(FFBB)
    
    for eachline in FFBB:
        p=eachline
        cv2.rectangle(image, tuple(p[0:2].astype(int)), tuple(p[0:2].astype(int) + p[2:4].astype(int)), (255,0,0), 2)
       
    fop.close()
    
    cv2.imwrite('/home/ran/Trails/HL60/firstframe.jpg',image)
    cv2.imshow('image',image)  
    
    return print('OK')
    

if __name__ == "__main__":
    image=cv2.imread('/home/ran/Trails/HL60/0001.jpg') #first frame jpg path
    bbox = DoG(image)
    drawfirstbbox(bbox)
    
    #return(FFBB)