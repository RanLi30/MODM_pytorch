# ------------------------------------------------------------------
# Tensorflow implementation of
#  "Learning Dynamic Memory Networks for Object Tracking", ECCV,2018
# Licensed under The MIT License [see LICENSE for details]
# Written by Tianyu Yang (tianyu-yang.com)
# ------------------------------------------------------------------

import collections
import tensorflow as tf

import config


AccessState = collections.namedtuple('AccessState', (
        'init_memory', 'memory', 'read_weight', 'write_weight', 'control_factors', 'write_decay', 'usage'))


def _reset_and_write(memory, write_weight, write_decay, control_factors, values):

    weight_shape = write_weight.get_shape().as_list()
    write_weight = tf.reshape(write_weight, weight_shape+[1,1,1])
    decay = write_decay*tf.expand_dims(control_factors[:, 1], 1) + tf.expand_dims(control_factors[:, 2], 1)
    decay_expand = tf.expand_dims(tf.expand_dims(tf.expand_dims(decay, 1), 2), 3)
    decay_weight = write_weight*decay_expand

    memory *= 1 - decay_weight
    values = tf.expand_dims(values, 1)
    memory += decay_weight * values

    return memory



def get_key_feature(inpu, name):
    input_shape = inpu.get_shape().as_list()
    if len(input_shape) > 4:
        inpu = tf.reshape(inpu, [-1] + input_shape[2:])
    contrloller_input = tf.layers.average_pooling2d(inpu, config.slot_size[0:2], [1, 1], 'valid', name=name) #!!!!!!!!!!!!!!!!!! avg_pooling
    if len(input_shape) > 4:
        c_shape = contrloller_input.get_shape().as_list()
        contrloller_input = tf.reshape(contrloller_input, input_shape[0:2]+c_shape[1:])
    return contrloller_input


def calc_allocation_weight(usage, memory_size):
    usage = tf.stop_gradient(usage)
    nonusage = 1 - usage
    sorted_nonusage, indices = tf.nn.top_k(nonusage, k=1, name='sort')
    allocation_weights = tf.one_hot(tf.squeeze(indices, [1]), memory_size)
    return allocation_weights


def update_usage(write_weights, read_weights, prev_usage):
    usage = config.usage_decay*prev_usage + write_weights + read_weights
    return usage

_EPSILON = 1e-6
def _vector_norms(m):
    squared_norms = tf.reduce_sum(m * m, axis=2, keep_dims=True)
    return tf.sqrt(squared_norms + _EPSILON)


def _weighted_softmax(activations, strengths, strengths_op):
    sharp_activations = activations * strengths_op(strengths)
    softmax_weights = tf.nn.softmax(sharp_activations)
    return softmax_weights


def cosine_similarity(memory, keys, strengths, strength_op=tf.nn.softplus):
    # Calculates the inner product between the query vector and words in memory.
    keys = tf.expand_dims(keys, 1)
    dot = tf.matmul(keys, memory, adjoint_b=True)
    # Outer product to compute denominator (euclidean norm of query and memory).
    memory_norms = _vector_norms(memory)
    key_norms = _vector_norms(keys)
    norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
    # Calculates cosine similarity between the query vector and words in memory.
    similarity = dot / (norm + _EPSILON)
    return _weighted_softmax(tf.squeeze(similarity, [1]), strengths, strength_op)


def attention_read(read_key, memory_key):
    memory_key = tf.expand_dims(memory_key, 1)
    input_transform = tf.layers.conv2d(memory_key, 256, [1, 1], [1, 1], use_bias=False, name='memory_key_layer')
    query_transform = tf.layers.dense(read_key, 256, name='read_key_layer')
    query_transform = tf.expand_dims(tf.expand_dims(query_transform, 1), 1)
    addition = tf.nn.tanh(input_transform + query_transform, name='addition_layer')
    addition_transform = tf.layers.conv2d(addition, 1, [1, 1], [1, 1], use_bias=False, name='score_layer')
    addition_shape = addition_transform.get_shape().as_list()
    return tf.nn.softmax(tf.reshape(addition_transform, [addition_shape[0], -1]))


class MemoryAccess(tf.nn.rnn_cell.RNNCell):

    def __init__(self, memory_size, slot_size, is_train):
        super(MemoryAccess, self).__init__()
        self._memory_size = memory_size
        self._slot_size = slot_size
        self._is_train = is_train


    def __call__(self, inputs, prev_state, scope=None):

        memory_for_writing = inputs[0]
        controller_output = inputs[1]
        read_key, read_strength, control_factors, write_decay, residual_vector = self._transform_input(controller_output)
        tf.print(read_strength)
        # Write previous template to memory.
        memory = _reset_and_write(prev_state.memory, prev_state.write_weight,
                                  prev_state.write_decay, prev_state.control_factors, memory_for_writing)

        # Read from memory.
        read_weight = self._read_weights(read_key, read_strength, memory)
        read_weight_expand = tf.reshape(read_weight, [-1, self._memory_size, 1, 1, 1])
        residual_vector = tf.reshape(residual_vector, [-1, 1, 1, 1, self._slot_size[2]])
        #read_memory = tf.reduce_sum(2*residual_vector*read_weight_expand*memory, [1])
        read_memory = tf.reduce_sum(read_weight_expand*memory, [1])
        
        # calculate the allocation weight
        allocation_weight = calc_allocation_weight(prev_state.usage, self._memory_size)

        # calculate the write weight for next frame writing
        write_weight = self._write_weights(control_factors, read_weight, allocation_weight)

        # update usage using read & write weights and previous usage
        usage = update_usage(write_weight, read_weight, prev_state.usage)

        # summary
        if int(scope) < config.summary_display_step:
            tf.summary.histogram('write_factor/{}'.format(scope), control_factors[:, 0])
            tf.summary.histogram('read_factor/{}'.format(scope), control_factors[:, 1])
            tf.summary.histogram('allocation_factor/{}'.format(scope), control_factors[:, 2])
            tf.summary.histogram('residual_vector/{}'.format(scope), residual_vector)
            tf.summary.histogram('write_decay/{}'.format(scope), write_decay)
            tf.summary.histogram('read_key/{}'.format(scope), read_key)
            if not config.use_attention_read:
                tf.summary.histogram('read_strength/{}'.format(scope), read_strength)

        return read_memory+0.5*prev_state.init_memory, AccessState(
        #return read_memory, AccessState(
            init_memory=prev_state.init_memory,
            memory=memory,
            write_weight=write_weight,
            read_weight=read_weight,
            control_factors=control_factors,
            write_decay=write_decay,
            usage=usage)

    def _transform_input(self, input):

        control_factors = tf.nn.softmax(tf.layers.dense(input, 3, name='control_factors'))
        write_decay = tf.sigmoid(tf.layers.dense(input, 1, name='write_decay'))
        residual_vector = tf.sigmoid(tf.layers.dense(input, self._slot_size[2], name='add_vector'))

        read_key = tf.layers.dense(input, config.key_dim, name='read_key')
        if config.use_attention_read:
            read_strength = None
        else:
            read_strength = tf.layers.dense(input, 1, bias_initializer=tf.ones_initializer(), name='write_strengths')

        return read_key, read_strength, control_factors, write_decay, residual_vector

    def _write_weights(self, control_factors, read_weight, allocation_weight):

        return tf.expand_dims(control_factors[:, 1], 1) * read_weight + tf.expand_dims(control_factors[:, 2], 1) * allocation_weight

    def _read_weights(self, read_key, read_strength, memory):

        memory_key = tf.squeeze(get_key_feature(memory,'memory_key'),[2,3])
        if config.use_attention_read:
            return attention_read(read_key, memory_key)
        else:
            return cosine_similarity(memory_key, read_key, read_strength)


    @property
    def state_size(self):

        return AccessState(init_memory=tf.TensorShape([self._memory_size]+self._slot_size),
                memory=tf.TensorShape([self._memory_size]+self._slot_size),
                read_weight=tf.TensorShape([self._memory_size]),
                write_weight=tf.TensorShape([self._memory_size]),
                write_decay=tf.TensorShape([1]),
                control_factors=tf.TensorShape([3]),
                usage=tf.TensorShape([self._memory_size]))

    @property
    def output_size(self):

        return tf.TensorShape(self._slot_size)
