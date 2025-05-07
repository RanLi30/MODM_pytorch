#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:50:43 2019

@author: ra


#detect for every frame

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#For every frame in sequence, detect the cell and using white bounding box. Output.

import time
import sys
sys.path.append('../')
import cv2
import numpy as np
from centroid import centerpoint
from bboxdetct import bboxdetc
from combinedbbox import compare2bbox

#file_handle=open('/home/ran/Trails/RFL-master/tracking/result1.txt', mode= 'w')
#def drawframe(image, pred_boxes, frame_idx, seq_name=None):    
#    for idx in range(1, len(s_frames)):
#       line = linecache.getline(file_handle, idx)  
#        bbox = np.fromstring(line)
#        display_result(cur_frame,bbox, idx)
#        res.append(bbox.tolist())
#    file_handle.close()
 


tn=1 #target Number
image_path_head   = "/home/ran/Trails/PhC-C2DH-U373/CODE/01/01jpg/"
#image_path_head   = "/home/ran/Trails/PhC-C2DH-U373/02T/img/"
#image_path_head   = "/home/ran/Data/exp_F0001 1-113/data/data/"
image_path_tail   = ".jpg"
image_path_extra  = "'"
'''
for p in range(1,tn+1):
        p=str(p)
   # fn1='/home/ran/Trails/RFL-master/tracking/result1.txt'
        #exec('fn' + p +"='/home/ran/Trails/RFL-master/tracking/NEWresult_c"+ p +".txt'")###### New result here
        exec('fn' + p +"='/home/ran/Trails/BF-C2DL-MuSC/Test/01/tracking/NEWresult"+ p +".txt'")
        #exec('fn' + p +"='/home/ran/Trails/MemTrack-master/tracking/02T/result"+ p +".txt'")
        
        
        exec('linenp'+p+'=np.loadtxt(fn'+p+')')
        exec('cp'+p+'=centerpoint(linenp'+p+')')
        exec('colo'+p+'=np.uint8(np.random.uniform(0, 255, 3))')
        exec('color'+p+' = tuple(map(int, colo'+p+'))')
        

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


start_time = time.time()
for idx in range(0,115):
    idx2 = idx
    idxname = "%04d" % idx2
    image_seq     = idxname
    image_path    = "%s%s%s"%(image_path_head,image_seq,image_path_tail)
    idx1 = idx
    #idx = "%04d" % idx1
    '''
    for q in range (1,tn+1):
        q=str(q)
        exec('pred_boxes'+ q +'=linenp'+ q +'[idx1]')
        exec('points'+ q +'=np.array(cp'+ q +'[1:idx1])')
        exec('pred_boxes'+ q +'=pred_boxes'+ q +'.astype(int)')
        
    pred_boxes1=linenp1[idx1]                         #bounding box coordinates based on txt file            
    ...   
    
    points1=np.array(cp1[1:idx1])                     #center points array
    ...
        
    pred_boxes1 = pred_boxes1.astype(int)             #coordinates flo change to int
    ...
    '''    
 
 #   colo1=colo1.astype(np.int32)
    #color1=tuple(colo1)
    # image=cv2.imread(image_path,cv2.IMREAD_COLOR)
    #r, g, b = cv2.split(image)
    #image = cv2.merge([b, g, r])
    image=cv2.imread(image_path)
    size=image.shape
    '''
    imageori= cv2.imread(image_path,cv2.IMREAD_COLOR) #original image
    ro,go,bo= cv2.split(imageori)
    imageori=cv2.merge([bo,go,ro])
    '''
    Bboxdect=bboxdetc(image)
    #Bboxdect=compare2bbox(Bboxdect)
    image1=np.zeros([size[0], size[1]], np.uint16)
    for bs in range (0,len(Bboxdect)):
        center=(int(Bboxdect[bs][0]+Bboxdect[bs][2]*0.5),int(Bboxdect[bs][1]+Bboxdect[bs][3]*0.5))
        image1=cv2.circle(image1, center, 10,(bs+1,bs+1,bs+1), -1)
        #cv2.rectangle(image, (int(Bboxdect[bs][0]),int(Bboxdect[bs][1])), (int(Bboxdect[bs][0]+Bboxdect[bs][2]),int(Bboxdect[bs][1]+Bboxdect[bs][3])), (255,255,255), 2)
    
    #for r in range (1,2):
        #r=str(r)
        #exec('cv2.rectangle(image, tuple(pred_boxes'+r+'[0:2]), tuple(pred_boxes'+r+'[0:2] + pred_boxes'+r+'[2:4]), color'+r+', 2)')
        #cv2.putText(image, 'Frame: %d' %idx1, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))                #frame indicator
    
        #exec('cv2.polylines(image, np.int32([points'+r+']), 0, color'+r+',2)   ')
    
    '''  
    cv2.rectangle(image, tuple(pred_boxes1[0:2]), tuple(pred_boxes1[0:2] + pred_boxes1[2:4]), (0,0,255), 2)       #draw bounding Box
    cv2.polylines(image, np.int32([points1]), 0, (0,0,255),2)                                                     #plot center point trajectory   
    cv2.rectangle(image, tuple(pred_boxes2[0:2]), tuple(pred_boxes2[0:2] + pred_boxes2[2:4]), (0, 255, 0), 2)
    cv2.polylines(image, np.int32([points2]), 0, (0,255,0),2)
    cv2.rectangle(image, tuple(pred_boxes3[0:2]), tuple(pred_boxes3[0:2] + pred_boxes3[2:4]), (255, 0, 0), 2)
    cv2.polylines(image, np.int32([points3]), 0, (255,0,0),2)
    cv2.rectangle(image, tuple(pred_boxes4[0:2]), tuple(pred_boxes4[0:2] + pred_boxes4[2:4]), (0, 255, 255), 2)
    cv2.polylines(image, np.int32([points4]), 0, (0,255,255),2)
    cv2.rectangle(image, tuple(pred_boxes5[0:2]), tuple(pred_boxes5[0:2] + pred_boxes5[2:4]), (255, 255, 0), 2)
    cv2.polylines(image, np.int32([points5]), 0, (255,255,0),2)
    cv2.rectangle(image, tuple(pred_boxes6[0:2]), tuple(pred_boxes6[0:2] + pred_boxes6[2:4]), (255, 0, 255), 2)
    cv2.polylines(image, np.int32([points6]), 0, (255,0,255),2)
    cv2.rectangle(image, tuple(pred_boxes7[0:2]), tuple(pred_boxes7[0:2] + pred_boxes7[2:4]), (0,0 ,0 ), 2)
    cv2.polylines(image, np.int32([points7]), 0, (0,0,0),2)
    cv2.rectangle(image, tuple(pred_boxes8[0:2]), tuple(pred_boxes8[0:2] + pred_boxes8[2:4]), (255,255 , 255), 2)
    cv2.polylines(image, np.int32([points8]), 0, (255,255,255),2)
    '''
    #cv2.putText(image, 'Frame: %d' %idx1, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))                #frame indicator
    #cv2.imshow('image',image)
    #cv2.putText(imageori, 'Frame: %d' %idx1, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))             #original image comparision
    #cv2.imshow('original',imageori)
    
    cv2.imwrite('/home/ran/Trails/PhC-C2DH-U373/CODE/01/detectforgt/'+str(idx)+'.tif', image1)
    #cv2.imwrite('/home/ran/Data/exp_F0001 1-113/result/Res/'+str(idx2)+'.jpg', image)
                                 #save the images
    #cv2.waitKey(0)
    #cv2.waitKey(0) 
    #cv2.destroyAllWindows()

end_time=time.time





















































#def readfile(filename):
#	with open(filename,'r') as f:
#		for line in f.readlines():
#			linestr = line.strip()
#			print (linestr)
#			linestrlist = linestr.split("\t")
#			print (linestrlist)
#			linelist = map(int,linestrlist)# 方法一
#			# linelist = [int(i) for i in linestrlist] # 方法二
#			print (linelist)
#  return linelist


   
    

