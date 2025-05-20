# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------


import tensorflow as tf
import collections
from tensorflow.python.util import nest

import config
import memnet


class ModeKeys():
  TRAIN = 'train'
  EVAL = 'eval'
  PREDICT = 'predict'

EstimatorSpec = collections.namedtuple('EstimatorSpec', ['predictions', 'loss', 'dist_error', 'train', 'summary', 'saver'])


# ------------------------------------------------------------------------------------------
def conv2d(input, filters, kernel_size, strides, padding, name, group=1):
    if group == 1:
        conv = tf.layers.conv2d(input, filters, kernel_size, strides, padding, name=name)
    else:
        input_group = tf.split(input, group, 3)
        conv_group = [tf.layers.conv2d(input, filters//group, kernel_size, strides, padding, name=name+'group_{}'.format(i))
                      for i, input in enumerate(input_group)]
        conv = tf.concat(conv_group, 3)
    return conv

def conv2d_bn_relu(is_train, input, filters, kernel_size, strides, padding, name, group=1):
    conv = conv2d(input, filters, kernel_size, strides, padding, name, group)
    bn = tf.layers.batch_normalization(conv, training=is_train, name=name+'_bn')
    return tf.nn.relu(bn, name=name+'_relu')

def extract_feature(is_train, img_patch):
    conv1 = conv2d_bn_relu(is_train, img_patch, 96, [11, 11], [2, 2], 'valid', name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1, [3, 3], [2, 2], 'valid', name='pool1')
    conv2 = conv2d_bn_relu(is_train, pool1, 256, [5, 5], [1, 1], 'valid', name='conv2')
    pool2 = tf.layers.max_pooling2d(conv2, [3, 3], [2, 2], 'valid', name='pool2')
    conv3 = conv2d_bn_relu(is_train, pool2, 384, [3, 3], [1, 1], 'valid', name='conv3')
    conv4 = conv2d_bn_relu(is_train, conv3, 384, [3, 3], [1, 1], 'valid', name='conv4')
    conv5 = tf.layers.conv2d(conv4, 256, [3, 3], [1, 1], 'valid', name='conv5')
    return conv5

def get_cnn_feature(inpu, reuse, mode):
    input_shape = inpu.get_shape().as_list()
    if len(input_shape) > 4:
        inpu = tf.reshape(inpu, [-1] + input_shape[2:])

    is_train = True if mode == ModeKeys.TRAIN else False
    
    with tf.variable_scope('feature_extraction', reuse=reuse):
        cnn_feature = extract_feature(is_train, inpu)

    if len(input_shape) > 4:
        cnn_feature_shape = cnn_feature.get_shape().as_list()
        cnn_feature = tf.reshape(cnn_feature, input_shape[0:2]+cnn_feature_shape[1:])
    
    return cnn_feature

# ------------------------------------------------------------------------------------------------

def batch_conv(A, B, mode):
    a_shape = A.get_shape().as_list()
    if len(a_shape) > 4:
        A = tf.reshape(A, [-1] + a_shape[2:])
    b_shape = B.get_shape().as_list()
    if len(b_shape) > 4:
        B = tf.reshape(B, [-1] + b_shape[2:])
    batch_size = A.get_shape().as_list()[0]
    output = tf.map_fn(lambda inputs: tf.nn.conv2d(tf.expand_dims(inputs[0], 0), tf.expand_dims(inputs[1], 3), [1,1,1,1], 'VALID'),
                       elems=[A, B],
                       dtype=tf.float32,
                       parallel_iterations=batch_size)
    is_train = True if mode == ModeKeys.TRAIN else False
    output = tf.layers.batch_normalization(tf.squeeze(output, [1]), training=is_train, name='bn_response')
    return tf.squeeze(output, [3])


def get_predictions(query_feature, search_feature, mode):
    with tf.variable_scope('mann'):
       mann_cell = memnet.MemNet(config.hidden_size, config.memory_size, config.slot_size, True)
    initial_state = mann_cell.initial_state(query_feature[:, 0])
    inputs = (search_feature, query_feature)
    outputs, final_state = rnn(cell=mann_cell, inputs=inputs, initial_state=initial_state)
    tf.print(search_feature)
    response = batch_conv(search_feature, outputs, mode)
    return response


def focal_loss(labels, predictions, gamma=2, epsilon=1e-7, scope=None):
    with tf.name_scope(scope, "focal_loss", (predictions, labels)) as scope:
        predictions = tf.to_float(predictions)
        labels = tf.to_float(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        preds = tf.where(
            tf.equal(labels, 1), predictions, 1. - predictions)
        losses = -(1. - preds) ** gamma * tf.log(preds + epsilon)
        return losses


def get_loss(outputs, labels, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return None
    # add code ...
    return None


def get_dist_error(outputs, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        return None
    outputs_shape = outputs.get_shape().as_list()
    outputs = tf.reshape(outputs, [outputs_shape[0], -1])
    pred_loc_idx = tf.argmax(outputs, 1)
    loc_x = pred_loc_idx%outputs_shape[1]
    loc_y = pred_loc_idx//outputs_shape[1]
    pred_loc = tf.stack([loc_x, loc_y], 1)
    gt_loc = tf.tile(tf.expand_dims([outputs_shape[1]/2, outputs_shape[1]/2], 0), [outputs_shape[0], 1])
    dist_error = tf.losses.mean_squared_error(predictions=pred_loc, labels=gt_loc)
    tf.summary.scalar('dist_error', dist_error)
    return dist_error


def get_train_op(loss, mode):
    if mode != ModeKeys.TRAIN:
        return None
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_circles, config.lr_decay, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    tvars = tf.trainable_variables()
    regularizer = tf.contrib.layers.l2_regularizer(config.weight_decay)
    regularizer_loss = tf.contrib.layers.apply_regularization(regularizer, tvars)
    loss += regularizer_loss
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.clip_gradients)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    batchnorm_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(batchnorm_update_ops):
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step)
    return train_op


def get_summary(mode):
    if mode == ModeKeys.PREDICT:
        return None
    return tf.summary.merge_all()


def get_saver():
    return tf.train.Saver(tf.global_variables(), max_to_keep=15)


def build_initial_state(init_query, mem_cell, mode):
    query_feature = get_cnn_feature(init_query, None, mode)
    return mem_cell.initial_state(query_feature[:,0])


def build_model(query, search, mem_cell, initial_state, mode):
    # get cnn feature for query and search
    query_feature = get_cnn_feature(query, True, mode)
    search_feature = get_cnn_feature(search, True, mode)
    inputs = (search_feature, query_feature)
    outputs, final_state = rnn(cell=mem_cell, inputs=inputs, initial_state=initial_state)
    response = batch_conv(search_feature, outputs, mode)
    saver = get_saver()
    return response, saver, final_state, outputs,query_feature,search_feature


# ---------------------------------------------------------------------------------------------
def weights_summay(weight, name):
    weight_shape = weight.get_shape().as_list()
    for i in range(weight_shape[1]):
        tf.summary.histogram('Memory_{}/{}'.format(i, name), weight[:,i])

def rnn(cell, inputs, initial_state, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`."""

    if not isinstance(cell, tf.contrib.rnn.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    input_shape = inputs[0].get_shape().as_list()
    input_list = []
    for input in inputs:
        input_list.append([tf.squeeze(input_, [1])
                           for input_ in tf.split(axis=1, num_or_size_splits=input_shape[1], value=input)])

    num_input = len(inputs)
    inputs = []
    for i in range(input_shape[1]):
        inputs.append(tuple([input_list[j][i] for j in range(num_input)]))
        
    outputs = []
    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        state = initial_state

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            #call_cell = lambda: cell(input_, state, str(time))
            call_cell = lambda: cell(input_,state, str(time))
            output, state = call_cell()
            outputs.append(output)
            states.append(state)

    # summary for all these weights
    if len(inputs) >= config.summary_display_step:
        for i in range(config.summary_display_step):
            state = states[i]
            weights_summay(state.access_state.memory, 'memory_slot/{}'.format(i))
            weights_summay(state.access_state.read_weight, 'read_weight/{}'.format(i))
            weights_summay(state.access_state.write_weight, 'write_weight/{}'.format(i))
            weights_summay(state.access_state.usage, 'usage/{}'.format(i))
    output_shape = outputs[0].get_shape().as_list()
    #outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, input_shape[1]] + output_shape[1:])
    outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, input_shape[1]] + output_shape[1:])
    print(outputs)
    return (outputs, state)
# -----------------------------------------------------------------------------------------------------------
