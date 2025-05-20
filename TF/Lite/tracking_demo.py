
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 15:15:50 2019

@author: ran
"""

import tensorflow as tf
import os
import cv2
import numpy as np
import scipy.spatial.distance

import config
import tracker


def load_init_bb(idx,cn):
    GTFile='groundtruth_rect'+str(cn)+'.txt'
    gt_file = open(os.path.join(config.otb_data_dir+config.code_seq,GTFile))

    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)
    return gt_rects[idx]    
    
    
def load_seq_fns():
    img_path = os.path.join(config.otb_data_dir+config.image_seq)
    img_names = sorted(os.listdir(img_path))
    seq_fns = [os.path.join(img_path, fn) for fn in img_names]
    return seq_fns


def display_result(image, pred_boxes, frame_idx, cp,seq_name=None):
    if len(image.shape) == 3:
        r, g, b = cv2.split(image)
        image = cv2.merge([b, g, r])
    pred_boxes = pred_boxes.astype(int)
    cv2.rectangle(image, tuple(pred_boxes[0:2]), tuple(pred_boxes[0:2] + pred_boxes[2:4]), (0, 0, 255), 2)

    cv2.putText(image, 'Frame: %d' % frame_idx, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    if config.is_save:
        cv2.imwrite(os.path.join(config.save_path,  '%04d.jpg' % frame_idx), image*255)


def clear():
    for key, value in globals().items():
        if callable(value) or value.__class__.__name__ == "module":
            continue
        del globals()[key]



if __name__ == '__main__':

    n_cells = 5
    
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        
        tracker = tracker.Tracker(tracker.Model(sess))

        for cn in range(1,n_cells+1):

            bbox0 = load_init_bb(0,cn)
            seq_fns = load_seq_fns()
            tracker.initialize(seq_fns[0], bbox0)

            file_save = open(os.path.join(config.otb_data_dir+config.code_seq,'CTRes'+str(cn)+'.txt'), mode='w')
            cplist=[]
            for idx in range(0, len(seq_fns)):
                if idx < 2:
                    redius = 0
                    #center = [math.floor(bbox[0]+0.5*bbox[2]),math.floor(bbox[1]+0.5*bbox[3])] #?????????????????????????
                    center = tracker.target_pos
                else:
                    center = [cplist[idx-1][0],cplist[idx-1][1]]
                    redius = scipy.spatial.distance.euclidean(cplist[idx-1],cplist[idx-2])

                #bbox, targetpos, cur_frame, response, instance,memory, templatenew = tracker.track(seq_fns[idx], redius, center, idx) #??????????????????????
                bbox, targetpos, cur_frame = tracker.track(seq_fns[idx], redius)
                
                cplist.append(targetpos)
                display_result(cur_frame, bbox, idx, targetpos)
                
                print('frame ',idx,':')
                print(bbox)
                
                bboxoutput = " ".join(str(i) for i in bbox.astype(np.int))
                file_save.write(bboxoutput)
                file_save.write('\n')
                
            file_save.close()

 
