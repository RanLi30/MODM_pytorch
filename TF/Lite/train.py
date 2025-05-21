import os
import numpy as np
import tensorflow as tf
import config
import memnet
import modelfuncs

# 1. 加载训练和验证数据（这里使用占位符和随机数据模拟）
query_ph = tf.placeholder(tf.float32, [None, 1, config.z_exemplar_size, config.z_exemplar_size, 3], name="query")
search_ph = tf.placeholder(tf.float32, [None, config.time_step, config.x_instance_size, config.x_instance_size, 3], name="search")
labels_ph = tf.placeholder(tf.float32, [None, config.time_step, config.response_size, config.response_size], name="labels")

# 2. 构建MemNet模型计算图
with tf.variable_scope('mann'):
    mem_cell = memnet.MemNet(config.hidden_size, config.memory_size, config.slot_size, is_train=True)
# 初始记忆状态：由第一帧（query）提取的特征初始化
initial_state = modelfuncs.build_initial_state(query_ph, mem_cell, modelfuncs.ModeKeys.TRAIN)
# 前向传播计算：得到响应图序列和其他返回值（包括模型保存器）
response, saver, final_state, outputs, query_feat, search_feat = modelfuncs.build_model(
    query_ph, search_ph, mem_cell, initial_state, modelfuncs.ModeKeys.TRAIN)

# 3. 定义损失函数和训练操作
loss = modelfuncs.get_loss(response, labels_ph, modelfuncs.ModeKeys.TRAIN)       # 计算预测响应与标签的损失
train_op = modelfuncs.get_train_op(loss, modelfuncs.ModeKeys.TRAIN)              # 获取训练优化操作（Adam优化器等）
dist_error = modelfuncs.get_dist_error(response, modelfuncs.ModeKeys.TRAIN)      # （可选）计算预测位置与中心的距离误差
summary_op = modelfuncs.get_summary(modelfuncs.ModeKeys.TRAIN)                   # 合并所有摘要（学习率、dist_error等）

# 为验证集定义单独的摘要（将验证损失写入TensorBoard）
val_loss_ph = tf.placeholder(tf.float32, shape=(), name="val_loss")
val_loss_summary = tf.summary.scalar('val_loss', val_loss_ph)

# 4. 训练循环设置
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True  # 按需分配GPU显存
with tf.Session(config=config_proto) as sess:
    sess.run(tf.global_variables_initializer())
    # 如果有预训练模型，加载预训练参数用于初始化（例如特征提取CNN的预训练权重）
    if os.path.isdir(config.pretrained_model_checkpoint_path):
        latest_pretrain = tf.train.latest_checkpoint(config.pretrained_model_checkpoint_path)
        if latest_pretrain:
            print("Restoring pretrained model from:", latest_pretrain)
            saver.restore(sess, latest_pretrain)
    # 如果已有检查点（断点续训），则恢复最新的检查点
    start_step = 0
    ckpt_state = tf.train.get_checkpoint_state(config.checkpoint_dir)
    if ckpt_state and ckpt_state.model_checkpoint_path:
        latest_ckpt = ckpt_state.model_checkpoint_path
        print("Resuming training from checkpoint:", latest_ckpt)
        saver.restore(sess, latest_ckpt)
        # 提取上次训练的step编号
        start_step = int(latest_ckpt.split('-')[-1])
    # 配置TensorBoard摘要记录
    summary_writer = tf.summary.FileWriter(config.summaries_dir, sess.graph)
    
    print("Starting training from step {}...".format(start_step))
    for step in range(start_step, config.max_iterations + 1):
        # 获取一个批次的训练数据（这里使用随机数据模拟）
        query_batch = np.random.rand(config.batch_size, 1, config.z_exemplar_size, config.z_exemplar_size, 3).astype(np.float32)
        search_batch = np.random.rand(config.batch_size, config.time_step, config.x_instance_size, config.x_instance_size, 3).astype(np.float32)
        # 构造标签：在response map中心位置为1，其他为0
        label_map = np.zeros((config.response_size, config.response_size), np.float32)
        center = config.response_size // 2
        label_map[center, center] = 1.0
        label_batch = np.tile(label_map, (config.batch_size, config.time_step, 1, 1))
        # 运行一次训练步骤，获取损失值
        _, train_loss = sess.run([train_op, loss], feed_dict={
            query_ph:  query_batch,
            search_ph: search_batch,
            labels_ph: label_batch
        })
        # 打印训练日志（每隔100步）
        if step % 100 == 0:
            print("Step {}: Training loss = {:.4f}".format(step, train_loss))
        # 保存训练摘要（如学习率、损失等，每隔summary_save_step步）
        if step % config.summary_save_step == 0:
            summary_str = sess.run(summary_op, feed_dict={
                query_ph:  query_batch,
                search_ph: search_batch,
                labels_ph: label_batch
            })
            summary_writer.add_summary(summary_str, step)
        # 每隔validate_step步，运行验证并记录日志和摘要
        if step % config.validate_step == 0 and step > 0:
            # 获取验证集批次（这里用随机数据模拟）
            val_query = np.random.rand(config.batch_size, 1, config.z_exemplar_size, config.z_exemplar_size, 3).astype(np.float32)
            val_search = np.random.rand(config.batch_size, config.time_step, config.x_instance_size, config.x_instance_size, 3).astype(np.float32)
            val_label = np.tile(label_map, (config.batch_size, config.time_step, 1, 1))
            # 计算验证损失
            val_loss = sess.run(loss, feed_dict={
                query_ph:  val_query,
                search_ph: val_search,
                labels_ph: val_label
            })
            print("Step {}: Validation loss = {:.4f}".format(step, val_loss))
            # 将验证损失写入TensorBoard摘要
            val_sum = sess.run(val_loss_summary, feed_dict={val_loss_ph: val_loss})
            summary_writer.add_summary(val_sum, step)
        # 每隔model_save_step步，保存模型检查点
        if step % config.model_save_step == 0 and step > 0:
            save_path = saver.save(sess, os.path.join(config.checkpoint_dir, "model.ckpt"), global_step=step)
            print("Model saved at {} (step {})".format(save_path, step))
    # 训练结束，关闭摘要记录器
    summary_writer.close()
    print("Training complete.")
