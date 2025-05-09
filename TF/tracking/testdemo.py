#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:33:33 2020

@author: ran
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:15:50 2019

@author: ran
"""

import tensorflow as tf
import time
import sys
sys.path.append('../')
import config
from tracking.tracker import Tracker, Model
import os
import cv2
from centroid import centerpoint
import numpy as np
from jumping import jumping

def load_seq_config(seq_name,trackidx,CellNum):
    
    GTFile='groundtruth_rect'+str(CellNum)+'.txt'
    
    src = os.path.join(config.otb_data_dir,GTFile)
    
    #src = os.path.join(config.otb_data_dir,seq_name,'groundtruth_rect1.txt')
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    init_rect= gt_rects[trackidx]
    img_path = os.path.join(config.otb_data_dir,seq_name,'adjust1')
    img_names = sorted(os.listdir(img_path))
    s_frames = [os.path.join(img_path, img_name) for img_name in img_names]

    return init_rect, s_frames

def display_result(image, pred_boxes, frame_idx, cp,seq_name=None):
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
    pred_boxes = pred_boxes.astype(int)
    points=np.array(cp[1:frame_idx])
    cv2.rectangle(image, tuple(pred_boxes[0:2]), tuple(pred_boxes[0:2] + pred_boxes[2:4]), (0, 0, 255), 2)

    cv2.putText(image, 'Frame: %d' % frame_idx, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))
    #cv2.polylines(image, np.int32([points]), 0, (255,255,255))
    cv2.imshow('tracker', image)
    #exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultinstance"+str(Frame_idx)+".jpg'),image)"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    if config.is_save:
        cv2.imwrite(os.path.join(config.save_path,  '%04d.jpg' % frame_idx), image)




def run_tracker(CelN):
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    
    ResFile='NEWresult'+str(CelN)+'.txt'
    file_save=open(os.path.join(config.otb_data_dir,'tracking/',ResFile), mode= 'w')
    #response_save=open('/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse24.txt', mode= 'w')
    fn=os.path.join(config.otb_data_dir,'tracking/',ResFile)
    #fnp24='/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse6.txt'
    linenp=np.loadtxt(fn)
    cp= centerpoint(linenp)
    

#for line in fr.readlines():
#    lineArr=line.strip().split('\t')
#    dataMat.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2])],float(lineArr[3]))

    #linenp=np.loadtxt(fn)   
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        os.chdir('../')
        model = Model(sess)
        tracker = Tracker(model)
        #######################################################
        trackidx=0
        #######################################################
        init_rect, s_frames = load_seq_config('',trackidx,CelN)
        
        bbox = init_rect
        res = []
        res.append(bbox)
        start_time = time.time()
        tracker.initialize(s_frames[0], bbox)
        for idx in range(trackidx, len(s_frames)):
        #for idx in range(trackidx, trackidx+100):
            if idx==551:
                bbox = [880,536,70,50]
            tracker.idx = idx
            bbox, cur_frame,response,instance= tracker.track(s_frames[idx])
            # #bboxw=np.array(bbox[0],bbox[1],bbox[2],bbox[3])
            ##print(bbox)
            
            bbox=bbox.astype(np.int)
            #print(" ".join(str(i) for i in bbox)) 
            bboxoutput=" ".join(str(i) for i in bbox)
            print(idx,':',bbox)
            #print(response)
           
            response_map=response.reshape(400,400)   #(272,272)        
            exec("np.savetxt(os.path.join(config.otb_data_dir,'tracking/resultresponse"+str(idx)+".txt'),response_map)")
            instance_map=instance.reshape(255,255,3)
            
            #print(np.shape(response))
            response_show=255/np.max(response_map)*response_map
            instance_show=255/np.max(instance_map)*instance_map
            #print(np.max(response_map))
            #exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultresponse"+str(idx)+".jpg'),response_show)")
            #exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultinstance"+str(idx)+".jpg'),instance_show)")
            #response_save.write('\n')
            
            #file_save.write(np.array2string(bbox))
            file_save.write(bboxoutput) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            file_save.write('\n')
            #bbox1=linenp
            display_result(cur_frame, bbox, idx,cp)
            res.append(bbox.tolist())
            time.sleep(0.001)
                        
        end_time = time.time()
        type = 'rect'
        fps = idx/(end_time-start_time)
        file_save.close()
        
    return res, type, fps

if __name__ == '__main__':      
        run_tracker(2)
