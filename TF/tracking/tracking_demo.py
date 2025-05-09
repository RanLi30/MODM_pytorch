
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
from tracking.tracker import Tracker, Model, memory_check
import matplotlib.pyplot as plt
import os
import cv2
from centroid import centerpoint
import numpy as np
from drawCTCRES import DrawCTCRES
import math




def load_seq_config(seq_name,trackidx,CellNum):
    
    GTFile='groundtruth_rect'+str(CellNum)+'.txt'
    
    src = os.path.join(config.otb_data_dir+config.code_seq,GTFile)
    
    #src = os.path.join(config.otb_data_dir,seq_name,'groundtruth_rect1.txt')
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    init_rect= gt_rects[trackidx]
    img_path = os.path.join(config.otb_data_dir+config.image_seq)
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
    #cv2.imshow('tracker', image)
    #exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultinstance"+str(Frame_idx)+".jpg'),image)"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    if config.is_save:
        cv2.imwrite(os.path.join(config.save_path,  '%04d.jpg' % frame_idx), image)

def clear():

    for key, value in globals().items():

        if callable(value) or value.__class__.__name__ == "module":

            continue

        del globals()[key]




def mkdir(path):

	folder = os.path.exists(path)

	if not folder:
		os.makedirs(path)

def run_tracker(lower,upper):

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        #keras.set_session(sess)
        os.chdir('../')
        model = Model(sess)
        tracker = Tracker(model)
        SPCHANGE=[]
        timeflag=0
        start_time = time.time()
        res = []
        for cn in range(lower,upper+1):
            #######################################################
            trackidx=0
            #######################################################

            init_rect, s_frames = load_seq_config('',trackidx,cn)

            ResFile='CTRes'+str(cn)+'.txt'

            file_save=open(os.path.join(config.otb_data_dir+config.code_seq,ResFile), mode= 'w')

            #response_save=open('/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse24.txt', mode= 'w')
            fn=os.path.join(config.otb_data_dir+config.code_seq,ResFile)
            #fnp24='/home/ran/Trails/N2DH-GOWH1/02/RESFILE/resultresponse6.txt'
            linenp=np.loadtxt(fn)
            cp= centerpoint(linenp)
            cplist=[]
            cplist.append(cp)
            rediuslist=[]



            bbox = init_rect
            init_center=[math.floor(bbox[0]+0.5*bbox[2]),math.floor(bbox[1]+0.5*bbox[3])]


            res.append(bbox)

            tracker.initialize(s_frames[0], bbox)
            response_dir=os.path.join(config.otb_data_dir+config.code_seq,'resultresponse/v1/'+str(cn))
            mkdir(response_dir)

            templatenorm=[]
            
            for idx in range(trackidx, len(s_frames)):

                tracker.idx = idx
                
                if idx < 2:
                    redius=0
                    center=init_center 
                else:
                    center=[cplist[idx][0],cplist[idx][1]]
                    P1_X=cplist[idx][0]
                    P1_Y=cplist[idx][1]
                    P2_X=cplist[idx-1][0]
                    P2_Y=cplist[idx-1][1]
                    
                    redius=math.sqrt((P1_X-P2_X)*(P1_X-P2_X)+(P1_Y-P2_Y)*(P1_Y-P2_Y))
                    rediuslist.append(redius)
                    
                    #print(np.mean(rediuslist))
                    #print('/')
                    #print(redius-np.mean(rediuslist))


                    SPCHANGE.append(redius)

                #bbox, cur_frame,response,instance= tracker.track(s_frames[idx])
                bbox,targetpos, cur_frame,response,instance,memory,templatenew= tracker.track(s_frames[idx],redius,center,idx)
                # #bboxw=np.array(bbox[0],bbox[1],bbox[2],bbox[3])
                #print(bbox)
                
                ############
                
                #if cn==5:####02seq
                #    if bbox[1]+bbox[3]<10:
                #       bbox[1]=bbox[1]-bbox[1]-bbox[3]+50
                    
                    
                    
                    
                ###############3    
                bbox=bbox.astype(np.int)
                res.append(bbox.tolist())
                cp0=[math.floor(bbox[0]+bbox[2]*0.5),math.floor(bbox[1]+bbox[3]*0.5)]
                cp=targetpos
                cplist.append(cp)
                
                                             
                
                
                #print(" ".join(str(i) for i in bbox)) 
                bboxoutput=" ".join(str(i) for i in bbox)
                
                
                print('frame ',idx,':')
                #print(bboxoutput)
                #print(cplist[idx])
                if idx >0:

                    print(res[idx])
                #print(response)
                response_shape=config.response_up*config.response_size
                response_map=response.reshape(response_shape,response_shape)   #(272,272)        
                #exec("np.savetxt(os.path.join(config.otb_data_dir,'tracking/resultresponse"+str(idx)+".txt'),response_map)")
                instance_map=instance.reshape(255,255,3)
                
                #print(np.shape(response))
                response_show=255/np.max(response_map)*response_map
                instance_show=255/np.max(instance_map)*instance_map
                #print(np.max(response_map))
                
                RESPFile=str(idx)+'.png'
    
                response_save = os.path.join(response_dir,RESPFile)
                
                exec("cv2.imwrite(response_save,response_show)")
                #exec("cv2.imwrite(os.path.join(config.otb_data_dir,'tracking/resultinstance"+str(idx)+".jpg'),instance_show)")
                #response_save.write('\n')
                
                #file_save.write(np.array2string(bbox))
                file_save.write(bboxoutput) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
                file_save.write('\n')
                #bbox1=linenp
                display_result(cur_frame, bbox, idx,cp) ###display results
                timeflag+=1
                time.sleep(0.01)
            
                if idx >= 1:
                    #plt.imshow(response_map)
                    #plt.show() 
                    #cv2.imwrite(response_map,'/home/ran/Desktop/1.png')
                    a=1
            
                if idx >= 8:
                    memory_check(memory,idx,trackidx)
                    x_norm = np.linalg.norm(templatenew)
                    #print(x_norm)

                    templatenorm.append(x_norm)

            type = 'rect'

            file_save.close()
        end_time = time.time()
        fps = timeflag/(end_time-start_time)
    return SPCHANGE,res,  fps


def speedanalysis(cn,SP):
        
    Speedtotal=[]
    Speedchangetotal=[]
    for cidx in range(0,cn):
        SPCHANGE=[]
        SPEEDORI=[]
        for i in range(0,int(len(SP)/cn)):
            
            fidx= int(i + cidx* len(SP)/cn)
               
            if i < 1:
                spchange=0
                
            else:
                spchange = SP[fidx]- SP[fidx-1]
            
            speedori=SP[fidx]
                
            #speed and change in current cell
            SPEEDORI.append(speedori)
            SPCHANGE.append(spchange)
            
            #speed and change among all cells
            Speedtotal.append(speedori)
            Speedchangetotal.append(spchange)
   
        exec("Speedfile='Speed"+str(cidx+1)+".txt'")
        speed_save=open(os.path.join(config.otb_data_dir+config.code_seq,Speedfile), mode= 'w')
        for speeds in SPEEDORI:
            speeds = str(speeds)
            speed_save.write(speeds) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speed_save.write('\n')
   
        exec("Speedchfile='Speedchange"+str(cidx+1)+".txt'")
        speedch_save=open(os.path.join(config.otb_data_dir+config.code_seq,Speedchfile), mode= 'w')
        for speedchs in SPCHANGE:#SPCHSANGE
            speedchs = str(speedchs)
            speedch_save.write(speedchs) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
            speedch_save.write('\n')

    return Speedtotal,Speedchangetotal


if __name__ == '__main__':
        cell_number=5   #################
        cn=cell_number

        SP,res, fps=run_tracker(1,cell_number)
        print(fps)

        #print(SPCHANGE)
        Speedtotal=[]
        Speedchangetotal=[]
        for cidx in range(0,cn):
            SPCHANGE=[]
            SPEEDORI=[]
            for i in range(0,int(len(SP)/cn)):
                
                fidx= int(i + cidx* len(SP)/cn)
                   
                if i < 1:
                    spchange=0
                    
                else:
                    spchange = SP[fidx]- SP[fidx-1]
                
                speedori=SP[fidx]
                    
                #speed and change in current cell
                SPEEDORI.append(speedori)
                SPCHANGE.append(spchange)
                
                #speed and change among all cells
                Speedtotal.append(speedori)
                Speedchangetotal.append(spchange)
   
            exec("Speedfile='Speed"+str(cidx+1)+".txt'")
            speed_save=open(os.path.join(config.otb_data_dir+config.code_seq,Speedfile), mode= 'w')
            for speeds in SPEEDORI:
                speeds = str(speeds)
                speed_save.write(speeds) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
                speed_save.write('\n')
   
            exec("Speedchfile='Speedchange"+str(cidx+1)+".txt'")
            speedch_save=open(os.path.join(config.otb_data_dir+config.code_seq,Speedchfile), mode= 'w')
            for speedchs in SPCHANGE:#SPCHSANGE
                speedchs = str(speedchs)
                speedch_save.write(speedchs) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
                speedch_save.write('\n')
                

        #DrawCTCRES(cell_number)
