# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import numpy as np
import tensorflow as tf
import cv2
import math

import config
import modelfuncs
import memnet
import access

class Model():

    def __init__(self, sess, checkpoint_dir=None):
    
        self.z_file_init = tf.placeholder(tf.string, [], name='z_filename_init') #占位符
        self.z_roi_init = tf.placeholder(tf.float32, [1, 4], name='z_roi_init')
        tmp,_ = self._read_and_crop_image(self.z_file_init, self.z_roi_init, [config.z_exemplar_size, config.z_exemplar_size])
        tmp = tf.reshape(tmp, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        init_z_exemplar = tf.tile(tmp, [config.num_scale, 1, 1, 1, 1])
        
        self.z_file = tf.placeholder(tf.string, [], name='z_filename')
        self.z_roi = tf.placeholder(tf.float32, [1, 4], name='z_roi')
        tmp,_ = self._read_and_crop_image(self.z_file, self.z_roi, [config.z_exemplar_size, config.z_exemplar_size])
        tmp = tf.reshape(tmp, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        z_exemplar = tf.tile(tmp, [config.num_scale, 1, 1, 1, 1])
        #self.z_exemplar=z_exemplar    %????????????????????????????????????????

        self.x_file = tf.placeholder(tf.string, [], name='x_filename')
        self.x_roi = tf.placeholder(tf.float32, [config.num_scale, 4], name='x_roi')
        self.x_instances, self.image = self._read_and_crop_image(self.x_file, self.x_roi, [config.x_instance_size, config.x_instance_size])
        self.x_instances = tf.reshape(self.x_instances, [config.num_scale, 1, config.x_instance_size, config.x_instance_size, 3])
        
        with tf.variable_scope('mann'):
            mem_cell = memnet.MemNet(config.hidden_size, config.memory_size, config.slot_size, False)   # !!!!!!!!!!!!!!!!!!!!!!!!!!
            
        self.initial_state = modelfuncs.build_initial_state(init_z_exemplar, mem_cell, modelfuncs.ModeKeys.PREDICT)  # !!!!!!!!!!!!!!!!!!!!!
        
        self.response, saver, self.final_state, self.outputs, self.query_feature, self.search_feature = modelfuncs.build_model(z_exemplar, 
                                                                                                                               self.x_instances, 
                                                                                                                               mem_cell, 
                                                                                                                               self.initial_state, 
                                                                                                                               modelfuncs.ModeKeys.PREDICT)
        
        self.att_score = mem_cell.att_score
        up_response_size = config.response_size * config.response_up
        self.up_response = tf.squeeze(tf.image.resize_images(tf.expand_dims(self.response,-1),
                                                             [up_response_size, up_response_size],method=tf.image.ResizeMethod.AREA,align_corners=True), -1)
        
        if checkpoint_dir is not None:
            saver.restore(sess, checkpoint_dir)
            self._sess = sess
        else:
            ckpt = tf.train.get_checkpoint_state(config.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                self._sess = sess

    def _read_and_crop_image(self, filename, roi, model_sz):
        image_file = tf.read_file(filename)
        # Decode the image as a JPEG file, this will turn it into a Tensor
        image = tf.image.decode_jpeg(image_file, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        frame_sz = tf.shape(image)
        # used to pad the crops
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')
        # pad with if necessary
        frame_padded, npad = self._pad_frame(image, frame_sz, roi, avg_chan)
        frame_padded = tf.cast(frame_padded, tf.float32)
        crop_patch = self._crop_image(frame_padded, npad, frame_sz, roi, model_sz)
        return crop_patch, image

    def _pad_frame(self, im, frame_sz, roi, avg_chan):
        pos_x = tf.reduce_max(roi[:, 0], axis=0)
        pos_y = tf.reduce_max(roi[:, 1], axis=0)
        patch_sz = tf.reduce_max(roi[:, 2:4], axis=0)
        c = patch_sz / 2
        xleft_pad = tf.maximum(0, -tf.cast(tf.round(pos_x - c[0]), tf.int32))
        ytop_pad = tf.maximum(0, -tf.cast(tf.round(pos_y - c[1]), tf.int32))
        xright_pad = tf.maximum(0, tf.cast(tf.round(pos_x + c[0]), tf.int32) - frame_sz[1])
        ybottom_pad = tf.maximum(0, tf.cast(tf.round(pos_y + c[1]), tf.int32) - frame_sz[0])
        npad = tf.reduce_max([xleft_pad, ytop_pad, xright_pad, ybottom_pad])
        paddings = [[npad, npad], [npad, npad], [0, 0]]
        im_padded = im
        if avg_chan is not None:
            im_padded = im_padded - avg_chan
        im_padded = tf.pad(im_padded, paddings, mode='CONSTANT')
        if avg_chan is not None:
            im_padded = im_padded + avg_chan
        return im_padded, npad

    def _crop_image(self, im, npad, frame_sz, rois, model_sz):
        radius = (rois[:, 2:4]-1) / 2
        c_xy = rois[:, 0:2]
        self.pad_frame_sz = pad_frame_sz = tf.cast(tf.expand_dims(frame_sz[0:2]+2*npad,0), tf.float32)
        npad = tf.cast(npad, tf.float32)
        xy1 = (npad + c_xy - radius)
        xy2 = (npad + c_xy + radius)
        norm_rect = tf.stack([xy1[:,1], xy1[:,0], xy2[:,1], xy2[:,0]], axis=1)/tf.concat([pad_frame_sz, pad_frame_sz],1)
        crops = tf.image.crop_and_resize(tf.expand_dims(im, 0), norm_rect, tf.zeros([tf.shape(rois)[0]],tf.int32), model_sz, method='bilinear')
        return crops




class Tracker():

    def __init__(self, model):
        self._model = model
        self._sess = model._sess
        #self.idx = 1
        # prepare constant things for tracking
        scale_steps=list(range(math.ceil(config.num_scale / 2) - config.num_scale, math.floor(config.num_scale / 2) + 1)) #通过scale数来生成一个矩阵储存不同的bbox变化scale
        self.scales = np.power(config.scale_multipler, scale_steps) #生成当前的bbox scale矩阵
        up_response_size = config.response_size * config.response_up #response放大 从17*17
        window = np.matmul(np.expand_dims(np.hanning(up_response_size), 1),np.expand_dims(np.hanning(up_response_size), 0)).astype(np.float32) #生成一个up_response 大小的mask,中心点是1
        self.window = window / np.sum(window) #归一化


    def estimate_bbox(self, responses, x_roi_size_origs, target_pos, target_size,radius):
        up_response_size = config.response_size * config.response_up
        current_scale_idx = math.floor(config.num_scale / 2) #向下取整
        best_scale_idx = current_scale_idx
        best_peak = -math.inf #负无穷
        for s_idx in range(config.num_scale):
            this_response = responses[s_idx].copy()
            this_peak = np.max(this_response)
            if this_peak > best_peak:
                best_peak = this_peak
                best_scale_idx = s_idx
        response = responses[best_scale_idx]    # ???????????????????????

        x_roi_size_orig = x_roi_size_origs[best_scale_idx]
        response = responses[current_scale_idx] # ???????????????????????
        
        #############################
        if 1: #frame_idx > 1: ??????????????????????
            mask_center=[0.5*response.shape[0],0.5*response.shape[1]]
            mask = gen_crater_mask(response, mask_center,config.radiusscale*radius/8, 2*config.sigma) # ?????????????
            mask_shr=cv2.resize(mask,(17,17),interpolation=cv2.INTER_AREA)
            response_shr= cv2.resize(response,(17, 17),interpolation=cv2.INTER_AREA)
            if config.MC==1:
                response = cv2.resize(mask_shr*response_shr,(response.shape[0], response.shape[1]),interpolation=cv2.INTER_LINEAR)
        #############################

        # make response sum to 1
        response -= np.min(response)
        response /= np.sum(response)
        #self.norm_response = response
        
        # apply window
        response = (1 - config.win_weights) * response + config.win_weights * self.window
        max_idx = np.argsort(response.flatten())
        max_idx = max_idx[-config.avg_num:]
        x = max_idx % up_response_size
        y = max_idx // up_response_size
        position = np.vstack([x, y]).transpose()
        shift_center = position - up_response_size // 2
        shift_center_instance = shift_center * config.stride / config.response_up
        shift_center_orig = shift_center_instance * np.expand_dims(x_roi_size_orig, 0) / config.x_instance_size
        target_pos_new = np.mean(target_pos + shift_center_orig, 0)
        target_size_new = target_size * self.scales[best_scale_idx]
        target_size = (1 - config.scale_damp) * target_size + config.scale_damp * target_size_new

        return target_pos_new, target_size, best_scale_idx


    def initialize(self, init_frame_file, init_box):
        self.pre_frame_file = init_frame_file
        bbox = np.array(init_box)
        self.target_pos = bbox[0:2] + bbox[2:4] / 2 #目标点位置
        self.target_size = bbox[2:4] #目标bbox大小

        self.z_roi_size = calc_z_size(self.target_size) #search region size ????????????????????????
        self.x_roi_size = calc_x_size(self.z_roi_size) #search region size  ????????????????????????
        
        self.next_state = self._sess.run(self._model.initial_state,
                                             {self._model.z_file_init: init_frame_file,
                                              self._model.z_roi_init: [np.concatenate([self.target_pos, self.z_roi_size], 0)]})
        

    def track(self, cur_frame_file, redius, display=False):
        sx_roi_size = np.round(np.expand_dims(self.x_roi_size, 0) * np.expand_dims(self.scales, 1))
        target_poses = np.tile(np.expand_dims(self.target_pos,axis=0), [config.num_scale,1])
        x_rois = np.concatenate([target_poses, sx_roi_size], axis=1)
        z_roi = np.concatenate([self.target_pos, self.z_roi_size], 0)
        att_score, responses, cur_frame, x_instances, self.next_state,Ftemplate, WITemplate, Search_region = self._sess.run([self._model.att_score,
                                                       self._model.up_response,
                                                       self._model.image,
                                                       self._model.x_instances,
                                                       self._model.final_state,
                                                       self._model.outputs,
                                                       self._model.query_feature,
                                                       self._model.search_feature],
                                                           {self._model.x_file: cur_frame_file,
                                                            self._model.x_roi: x_rois,
                                                            self._model.z_file: self.pre_frame_file,
                                                            self._model.z_roi: [z_roi],
                                                            self._model.initial_state: self.next_state})

        # estimate position and size
        Final_template=Ftemplate[0][0]
        self.target_pos, self.target_size, best_scale_idx = self.estimate_bbox(responses, sx_roi_size, self.target_pos, self.target_size,redius)
        bbox = np.hstack([self.target_pos - self.target_size / 2, self.target_size])
        
        self.next_state, memory = get_new_state(self.next_state, best_scale_idx)
        
        # calculate new x and z roi size for next frame
        self.z_roi_size = calc_z_size(self.target_size)
        self.x_roi_size = calc_x_size(self.z_roi_size)
        self.pre_frame_file = cur_frame_file
        
        return bbox, self.target_pos,cur_frame



# ????????????????????
def calc_z_size(target_size):
    # calculate roi region
    extend_size = target_size + config.context_amount * (target_size[0] + target_size[1])
    z_size = np.sqrt(np.prod(extend_size)) #extend_size长乘以宽的平方
    z_size = np.repeat(z_size, 2, 0) #size复制两倍
    return z_size


def calc_x_size(z_roi_size):
    # calculate roi region
    z_scale = config.z_exemplar_size / z_roi_size
    delta_size = config.x_instance_size - config.z_exemplar_size
    x_size = delta_size / z_scale + z_roi_size
    return x_size


def get_new_state(state, best_scale):
    lstm_state = state[0]
    access_state = state[1]
    c_best = lstm_state[0][best_scale]
    h_best = lstm_state[1][best_scale]
    c = np.array([c_best]*config.num_scale)
    h = np.array([h_best]*config.num_scale)
    lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
    s_list = []
    for s in access_state:
        s_best = s[best_scale]
        s_list.append([s_best]*config.num_scale)
    access_state = access.AccessState(np.array(s_list[0]), np.array(s_list[1]), np.array(s_list[2]),
                               np.array(s_list[3]), np.array(s_list[4]), np.array(s_list[5]), np.array(s_list[6]))
    memory=access_state[1][0]
    return memnet.MemNetState(lstm_state, access_state),memory


def gen_crater_mask(image, cpos, radius, sigma):
    shape=image.shape
    y,x = np.ogrid[1:shape[0]+1,1:shape[1]+1]
    x = x-cpos[0]
    y = y-cpos[1]
    dist = np.sqrt(x*x+y*y)
    h = np.exp(-(dist-radius)**2/(2.*sigma**2))
     
    maxh = h.max()
    if maxh != 0:
        h /= maxh

    eps = config.eps
    h[h<eps] = eps

    h2 =cv2.resize(h,(17,17),interpolation=cv2.INTER_AREA) 
    i2= cv2.resize(image,(17, 17),interpolation=cv2.INTER_AREA) 

    r = (i2 * h2).clip(0, 255).astype(np.uint8)
    r2= cv2.resize(r,(shape[0], shape[1]),interpolation=cv2.INTER_CUBIC) 
    return r2
