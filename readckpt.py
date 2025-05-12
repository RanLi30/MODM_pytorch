import tensorflow as tf

ckpt_path = "model.ckpt-60000"  # 

reader = tf.train.NewCheckpointReader(ckpt_path)
var_to_shape_map = reader.get_variable_to_shape_map()

print("=== TF checkpoint 内容如下：===\n")
for key in sorted(var_to_shape_map.keys()):
    print(f"{key} : shape {var_to_shape_map[key]}")
