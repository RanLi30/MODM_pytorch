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

def load_seq_config(seq_name,cellindex):
    loc = locals()
    #src = os.path.join(config.otb_data_dir,seq_name,'firstframe1.txt')
    filename='firstframe%s.txt'%cellindex
    src = os.path.join(config.otb_data_dir,seq_name,filename)
    #src=loc['src']
    gt_file = open(src)
    lines = gt_file.readlines()
    gt_rects = []
    for gt_rect in lines:
        rect = [int(v) for v in gt_rect[:-1].split(',')]
        gt_rects.append(rect)

    init_rect= gt_rects[0]
    img_path = os.path.join(config.otb_data_dir,seq_name,'data')
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    if config.is_save:
        cv2.imwrite(os.path.join(config.save_path, seq_name, '%04d.jpg' % frame_idx), image)

def run_tracker():
    loc=locals()
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    #exec('cellindex1 = cellindex')
    #cellindex1 = loc['cellindex1']
    '''
    dataMat=[];
    fr=open('/home/ran/Trails/MemTrack-master/tracking/result1.txt')
    fn='/home/ran/Trails/MemTrack-master/tracking/result1.txt'
    '''
    '''
    filename='/home/ran/Trails/MemTrack-master/tracking/02T/result%s.txt'%cellindex
    file_save=open(filename, mode= 'w')
    loc['fn'+str(cellindex)]='/home/ran/Trails/MemTrack-master/tracking/02T/result%s.txt'%cellindex
    loc['linenp'+str(cellindex)]=np.loadtxt(loc['fn'+str(cellindex)]) 
    cp= centerpoint(loc['linenp'+str(cellindex)])
    '''
#for line in fr.readlines():
#    lineArr=line.strip().split('\t')
#    dataMat.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2])],float(lineArr[3]))

    #linenp=np.loadtxt(fn)   
    with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:
        os.chdir('../')
        model = Model(sess)
        tracker = Tracker(model)
        for cellindex in range(25,26):
            filename='/home/ran/Data/exp_F0001 1-113/result/result%s.txt'%cellindex
            file_save=open(filename, mode= 'w')
            loc['fn'+str(cellindex)]='/home/ran/Data/exp_F0001 1-113/result/result%s.txt'%cellindex
            loc['linenp'+str(cellindex)]=np.loadtxt(loc['fn'+str(cellindex)]) 
            cp= centerpoint(loc['linenp'+str(cellindex)])

            init_rect, s_frames = load_seq_config('data',cellindex)
            bbox = init_rect
            res = []
            res.append(bbox)
            start_time = time.time()
            tracker.initialize(s_frames[0], bbox)
    
            for idx in range(0, len(s_frames)):
                tracker.idx = idx
                bbox, cur_frame = tracker.track(s_frames[idx])
                # bboxw=np.array(bbox[0],bbox[1],bbox[2],bbox[3])
                bbox=bbox.astype(np.int)
                print(" ".join(str(i) for i in bbox)) 
                bboxoutput=" ".join(str(i) for i in bbox)
                print(bboxoutput)
            
                #file_save.write(np.array2string(bbox))
                file_save.write(bboxoutput) #bboxoutput-output as 12 13 13 13 str(bbox)-output as [12 13 13 13]
                file_save.write('\n')
                #bbox1=linenp
                display_result(cur_frame, bbox, idx,cp)
                res.append(bbox.tolist())
                time.sleep(0.1)
                
            end_time = time.time()
            type = 'rect'
            fps = idx/(end_time-start_time)
            file_save.close()
    return res, type, fps

if __name__ == '__main__':
  
        run_tracker()
