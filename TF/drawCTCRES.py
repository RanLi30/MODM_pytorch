#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:44:28 2019

@author: ran
"""

import sys
sys.path.append('../')
import cv2
import numpy as np
from centroid import centerpoint
from PIL import Image
import os
import config

def DrawCTCRES(tn):
#image_path_head   = "/home/ran/Trails/MemTrack-master/results/02T/RES/"
#image_path_head   = "/home/ran/Trails/PhC-C2DH-U373/ideo/"
    image_path_head   = os.path.join(config.otb_data_dir+config.image_seq)
    image_path_tail   = ".jpg"
    image_path_extra  = "'"
    image_exp_path = "%s%s%s"%(image_path_head,'0000',image_path_tail)
    image_exp=cv2.imread(image_exp_path)
    size=image_exp.shape
    
    res_path_head = os.path.join(config.otb_data_dir+config.res_seq)
    res_path_tail = ".tif" 
    res_path_extra = "'" 
    
    for p in range(1,tn+1):
            p=str(p)
       # fn1='/home/ran/Trails/RFL-master/tracking/result1.txt'
            #exec('fn' + p +"='/home/ran/Trails/RFL-master/tracking/NEWresult_c"+ p +".txt'")###### New result here
            #exec('fn' + p +"='/home/ran/Trails/MemTrack-master/tracking/01T/RES6.25/result"+ p +".txt'")
            u=256/tn
            #ResFile='CTRes'+p+'.txt'
            #fname='fn'+p
            fpath = os.path.join(config.otb_data_dir+config.code_seq)
    
            
            exec('fn' + p +'=fpath+"CTRes'+ p +'.txt"')
            exec('linenp'+p+'=np.loadtxt(fn'+p+')')
            exec('linenp'+p+'=linenp'+p+'.astype(int)')
            exec('cp'+p+'=centerpoint(linenp'+p+')')
            '''
            exec('colo'+p+'=np.uint8(np.random.uniform(0, 255, 3))')
            exec('color'+p+' = tuple(map(int, colo'+p+'))')
            '''
            #u=256/tn
            exec('color'+p+'=(u*'+p+',u*'+p+',u*'+p+')')
    '''
    fn1='/home/ran/Trails/RFL-master/tracking/result1.txt' #load txt file
    ...
    linenp1=np.loadtxt(fn1)                                #read from txt file
    ...   
    cp1=centerpoint(linenp1)                               #centerpoint of boundingbox
    ...
    colo1=np.uint8(np.random.uniform(0,255,3))             #random 3-channel color
    ...
    color1=tuple(map(int.colo1))                           #convert to int
    ...
    '''
    
    background = np.zeros([size[0], size[1]], np.uint16)
    #image=cv2.imread(image_path,cv2.IMREAD_COLOR)
    
    
        #image=cv2.imread(image_path,cv2.IMREAD_COLOR)
    
    
        #cv2.imwrite('/home/ran/Trails/MemTrack-master/results/01T/RES/'+str(idx)+'.jpg', background)
    
    for idx in range(0,len(os.listdir(image_path_head))):
        idxname = "%03d" % idx
        res_seq     = 'mask'+idxname
        res_path    = "%s%s%s"%(res_path_head,res_seq,res_path_tail)
    ################   Height,Weidth  ###############33    
        image= np.zeros([520,696], np.uint16)
        #image= np.zeros([520, 696], np.uint16)
        #image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for q in range (1,tn+1):
            nq=q
            q=str(q)
            exec('radius'+q+'=0.2*linenp'+q+'[idx][2]')
            #exec('cv2.circle(image, (cp'+ q +'[idx][0],cp'+ q +'[idx][1]), 10, nq, -1)') #int(radius'+q+')
            #exec('cv2.rectangle(image, (linenp'+q+'[idx][0],linenp'+q+'[idx][1]), (linenp'+q+'[idx][0]+linenp'+q+'[idx][2],linenp'+q+'[idx][1]+linenp'+q+'[idx][3]), nq, -1)')
            exec('cv2.rectangle(image, (linenp'+q+'[idx][0],linenp'+q+'[idx][1]), (linenp'+q+'[idx][0]+linenp'+q+'[idx][2],linenp'+q+'[idx][1]+linenp'+q+'[idx][3]), nq, -1)')#for U373 +110
        #image.dtype=('unit16')
        image.dtype=np.uint16
        #d[()] = np.arange(200).reshape(10, 20)
        #exec("imsave('/home/ran/Trails/MemTrack-master/results/01T/RES7.30/mask"+idx1+".tif',image)")
        #exec("imsave('/home/ran/Trails/PhC-C2DH-U373/RESULTS/TE_S1_RES/V1/mask"+idx1+".tif',image)")
    
        #RESFile='mask'+idxname+'.tif' 
        #RES_save = os.path.join(res_path_head,RESFile)
        cv2.imwrite(res_path, image)
    
        ###cv2.imwrite('/home/ran/Trails/MemTrack-master/results/01T/RES/RESNEW/'+idx1+'.jpg', image) 
            
            
            #exec('pred_boxes'+ q +'=linenp'+ q +'[idx1]')
            ##exec('points'+ q +'=np.array(cp'+ q +'[1:idx1])')
            #exec('pred_boxes'+ q +'=pred_boxes'+ q +'.astype(int)')
            #img=data.chelsea()
            #exec('rr, cc=draw.circle(cp'+q+'[idx1][1],cp'+q+'[idx1][0],10)')
            #draw.set_color(image,[rr,cc],[255,255,255])
            #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #plt.imsave(image,plt.cm.gray)
    '''
    img=cv2.imread('/home/ran/Trails/MemTrack-master/results/02T/RES/0000.jpg')
    rr, cc=draw.circle(cp1[0][1],cp1[0][0],10)
    draw.set_color(img,[rr,cc],[255,255,255])
    plt.imshow(img,plt.cm.gray)        
    '''
    return(print('OK'))
if __name__ == '__main__':      
        DrawCTCRES(5)