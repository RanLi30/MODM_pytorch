import time
import sys

sys.path.append('../')
import cv2
import numpy as np
from centroid import centerpoint

# file_handle=open('/home/ran/Trails/RFL-master/tracking/result1.txt', mode= 'w')
# def drawframe(image, pred_boxes, frame_idx, seq_name=None):
#    for idx in range(1, len(s_frames)):
#       line = linecache.getline(file_handle, idx)
#        bbox = np.fromstring(line)
#        display_result(cur_frame,bbox, idx)
#        res.append(bbox.tolist())
#    file_handle.close()


src ="/home/ran/Trails/germination/no_germination_merged/CODE/01/firstframe1.txt"
gt_file = open(src)
lines = gt_file.readlines()
bbox = []
image = cv2.imread('/home/ran/Trails/germination/no_germination_merged/CODE/01/ori/0001.jpg')
r = 0
for gt_rect in lines:
    pred_boxes = [int(v) for v in gt_rect[:-1].split(',')]
    pred_boxes = np.array(pred_boxes)
    colo=np.uint8(np.random.uniform(0, 255, 3))

    color = tuple(map(int, colo))
    r = r + 1
    cv2.rectangle(image, tuple(pred_boxes[0:2]), tuple(pred_boxes[0:2] + pred_boxes[2:4]), (0,0,255), 2)       #draw bounding Box


cv2.imwrite('/home/ran/Trails/germination/no_germination_merged/CODE/01/result/detec.jpg', image)




