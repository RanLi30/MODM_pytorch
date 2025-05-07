import tensorflow as tf
import deepdish as dd
import argparse
import os
import numpy as np

def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().iteritems()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights

if __name__ == '__main__':
    outfile = '/home/ran/Trails/MemTrack-master/output/models/'

    weights = read_ckpt('/home/ran/Trails/MemTrack-master/output/models/model.ckpt-60000.data-00000-of-00001')
    dd.io.save(outfile, weights)
    weights2 = dd.io.load(outfile)