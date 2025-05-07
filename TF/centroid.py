#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:07:01 2018

@author: ran
"""
#import tensorflow as tf
import sys
sys.path.append('../')
#from tracking.rfl_tracker import RFLTracker, RFLearner
import numpy as np
import math

#fn1='/home/ran/Trails/RFL-master/tracking/result1.txt'
file_handle=open('/home/ran/Trails/MemTrack-master/tracking/center1.txt', mode= 'w')

#linenp1=np.loadtxt(fn1) 
def centroid(i,bbox):
    
    #bbox=bbox.astype(int)
    #centroid=np.array((bbox[i,0]+bbox[i,2]/2,bbox[i,1]+bbox[i,3]/2))
    centroid_x=math.floor(bbox[i,0]+bbox[i,2]/2)
    centroid_y=math.floor(bbox[i,1]+bbox[i,3]/2)
    return centroid_x,centroid_y

def centerpoint(linenp1):
    center=[]
    t=[]
    a=[]
    for idx in range(len(linenp1)):

#        t=centroid(idx,linenp1)  
        x,y=centroid(idx,linenp1)
        x=math.floor(x)
        y=math.floor(y)
        t=[x,y]
        #t=t.astype(int)
        center.append(t)
        
        a=np.array(center)
        a=a.astype(int)
        print(a)
        #file_handle.write(np.array2string(a))
        #file_handle.write('\n')
        #print(a)
#      t.append(t)
     #tbbox=tuple(linenp1[idx])
    return a

