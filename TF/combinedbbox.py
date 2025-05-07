#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:28:20 2019

@author: ran
"""
#combine all detected bounding boxes when overlapping with each other. 


import numpy as np
#from bboxdetct import bboxdetc



def compare2bbox (bboxlist):
    flag=1
    while flag>0:
        for i in range(0,len(bboxlist)):
            flag=0
            for j in range(i+1,len(bboxlist)):
                f=detectoverlap(bboxlist[i],bboxlist[j])
                if f>0:
                    #print('overlaped:',bboxlist[i], 'and', bboxlist[j])
                    c=combinedbbox(bboxlist[i],bboxlist[j])
                    #print(c)
                    bboxlist[i]=c
                    bboxlist[j]=[0,0,0,0]
                    flag=1
                    #bboxlist=np.delete(bboxlist,j,0)
            
        dindex=[]
        for p in range(0,len(bboxlist)):
            
            if bboxlist[p][3]<0:
                flag=1
                dindex.append(p)
        bboxlist=np.delete(bboxlist,dindex,0)               
    return bboxlist
    
    

    
def detectoverlap(bbox1,bbox2):
    A=[bbox1[0]-5,bbox1[1]-5,bbox1[0]+bbox1[2]+5,bbox1[1]+bbox1[3]+5]
    B=[bbox2[0]-5,bbox2[1]-5,bbox2[0]+bbox2[2]+5,bbox2[1]+bbox2[3]+5]
    iou=0
    iw = min(A[2], B[2]) - max(A[0], B[0])
    if iw > 0:
       ih = min(A[3], B[3]) - max(A[1], B[1])  
       if ih > 0:
            A_area = (A[2] - A[0]) * (A[3] - A[1])
            B_area = (B[2] - B[0]) * (B[3] - B[1])
            uAB = float(A_area + B_area - iw * ih)
            iou = iw * ih / uAB

    return iou


def combinedbbox(bbox1,bbox2):
    bbox=[min(bbox1[0],bbox2[0]),min(bbox1[1],bbox2[1]),max(bbox1[0]+bbox1[2],bbox2[0]+bbox2[2])-min(bbox1[0],bbox2[0]),max(bbox1[1]+bbox1[3],bbox2[1]+bbox2[3])-min(bbox1[1],bbox2[1])]
    return bbox

