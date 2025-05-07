import numpy as np
import tensorflow as tf
import sys
import cv2
import os
import config
from model import build_initial_state, build_model, ModeKeys
from memnet.memnet import MemNet, AccessState, MemNetState
import math
import matplotlib.pyplot as plt
from gen_crater_mask import gen_crater_mask
sys.path.append('../')
import tensorflow.keras.layers as KL


class Modelread():

    def __init__(self, sess, checkpoint_dir=None):

        self.z_file_init = tf.placeholder(tf.string, [], name='z_filename_init') #占位符
        self.z_roi_init = tf.placeholder(tf.float32, [1, 4], name='z_roi_init')
        self.z_file = tf.placeholder(tf.string, [], name='z_filename')
        self.z_roi = tf.placeholder(tf.float32, [1, 4], name='z_roi')
        self.x_file = tf.placeholder(tf.string, [], name='x_filename')
        self.x_roi = tf.placeholder(tf.float32, [config.num_scale, 4], name='x_roi')
        init_z_exemplar,_ = self._read_and_crop_image(self.z_file_init, self.z_roi_init, [config.z_exemplar_size, config.z_exemplar_size])
        init_z_exemplar = tf.reshape(init_z_exemplar, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        init_z_exemplar = tf.tile(init_z_exemplar, [config.num_scale, 1, 1, 1, 1])
        z_exemplar,_ = self._read_and_crop_image(self.z_file, self.z_roi, [config.z_exemplar_size, config.z_exemplar_size])
        z_exemplar = tf.reshape(z_exemplar, [1, 1, config.z_exemplar_size, config.z_exemplar_size, 3])
        z_exemplar = tf.tile(z_exemplar, [config.num_scale, 1, 1, 1, 1])
        self.x_instances, self.image = self._read_and_crop_image(self.x_file, self.x_roi, [config.x_instance_size, config.x_instance_size])
        self.x_instances = tf.reshape(self.x_instances, [config.num_scale, 1, config.x_instance_size, config.x_instance_size, 3])
        self.z_exemplar=z_exemplar




        with tf.variable_scope('mann'):
            mem_cell = MemNet(config.hidden_size, config.memory_size, config.slot_size, False)
        self.initial_state = build_initial_state(init_z_exemplar, mem_cell, ModeKeys.PREDICT)
        self.response, saver, self.final_state, self.outputs, self.query_feature,self.search_feature= build_model(z_exemplar, self.x_instances, mem_cell, self.initial_state, ModeKeys.PREDICT)
        query_feature=self.query_feature
        self.att_score = mem_cell.att_score
        up_response_size = config.response_size * config.response_up
        self.up_response = tf.squeeze(tf.image.resize_images(tf.expand_dims(self.response, -1),
                                                             [up_response_size, up_response_size],
                                                             method=tf.image.ResizeMethod.AREA,
                                                             align_corners=True), -1)

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
        radius = (rois[:, 2:4] - 1) / 2
        c_xy = rois[:, 0:2]
        self.pad_frame_sz = pad_frame_sz = tf.cast(tf.expand_dims(frame_sz[0:2] + 2 * npad, 0), tf.float32)
        npad = tf.cast(npad, tf.float32)
        xy1 = (npad + c_xy - radius)
        xy2 = (npad + c_xy + radius)
        norm_rect = tf.stack([xy1[:, 1], xy1[:, 0], xy2[:, 1], xy2[:, 0]], axis=1) / tf.concat(
            [pad_frame_sz, pad_frame_sz], 1)
        crops = tf.image.crop_and_resize(tf.expand_dims(im, 0), norm_rect, tf.zeros([tf.shape(rois)[0]], tf.int32),
                                         model_sz, method='bilinear')
        return crops

config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config_proto) as sess:

    os.chdir('../')
    model = Modelread(sess)
    a = 1
    print(sess.run(model.att_score))
