#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:46:37 2019

@author: ran
"""

import sys
sys.path.append('../')
import cv2
import numpy as np
from centroid import centerpoint
file_save=open('/home/ran/Trails/MemTrack-master/tracking/01T/RES6.25/result8.txt',mode='w')

tn=1
IniV=0
for p in range(3,4):
    p=str(p)
    #exec('fn' + p +"='/home/ran/Trails/MemTrack-master/tracking/01T/NEWresult"+ p +".txt'")
    exec('fn' + p +"='/home/ran/Trails/MemTrack-master/tracking/01T/RES6.25/result"+ p +".txt'")
    exec('linenp'+p+'=np.loadtxt(fn'+p+')')
    exec('cp'+p+'=centerpoint(linenp'+p+')')   
    V=0    
    
    for idx in range(0,len(linenp3)-1):
        
        linenp3[idx][0]=linenp3[idx][0]-20
        linenp3[idx][1]=linenp3[idx][1]-20
linenp4=str(linenp3)   
file_save.write(linenp4)
        
    

