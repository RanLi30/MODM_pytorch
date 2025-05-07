import tensorflow as tf
import sys

trained_checkpoint_prefix = sys.argv[1]
export_dir = sys.argv[2]
graph = tf.Graph()
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.compat.v1.Session(graph=graph, config=config) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph('model.ckpt-60000.data-00000-of-00001' + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('/home/ran/Trails/MemOri/MemTrack-master/output/models/covert')
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()