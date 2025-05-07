#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:00:54 2018

@author: ran
"""

import linecache
import numpy as np   

file_handle=open('/home/ran/Trails/RFL-master/tracking/result1.txt', mode= 'w')
def drawframe(image, pred_boxes, frame_idx, seq_name=None):    
    for idx in range(1, len(s_frames)):
        line = linecache.getline(file_handle, idx)  
        bbox = np.fromstring(line)
        display_result(cur_frame,bbox, idx)
        res.append(bbox.tolist())
    file_handle.close()
    