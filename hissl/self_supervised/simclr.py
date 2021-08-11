# encoding: utf-8

from absl import logging
import time
import os
import tensorflow as tf

from hissl.self_supervised.losses.contrastive_loss import simclr_contrastive_loss
from hissl.self_supervised.utils.network_utils import add_weight_decay


def self_supervised_training(network: tf.keras.models.Model,
                             train_dist_dataset,
                             train_data_len: int = 0,
                             batch_size: int = 32,
                             epochs: int = 10,
                             optimizer=tf.keras.optimizers.Adam(),
                             temperature: float = 0.1,
                             save_model: bool = True,
                             model_file: str = 'simclr_self_supervised_model.h5',
                             strategy: tf.distribute.Strategy = tf.distribute.MirroredStrategy()
                             ):
    """
    SimCLR 对比学习训练模块
    :param strategy: 分布式训练策略
    :param network: 特征提取网络
    :param train_db: 训练数据集，TFRecord格式
    :param train_data_len: 训练数据集长度
    :param batch_size: 批次
    :param epochs: 训练轮数
    :param optimizer: 优化器
    :param weight_path: 模型权重路径
    :param temperature: 温度，超参数，需要人工调整，非常关键的参数
    :param LARGE_NUM:
    :param model_file: 保存模型的名称
    :return:
    """
    if network is None:
        return

    GLOBAL_BATCH_SIZE = batch_size

    with strategy.scope():
        # 权重衰减Loss、总Loss记录
        weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
        total_loss_metric = tf.keras.metrics.Mean('train/total_loss')

        # 对比学习Loss
        def compute_loss(hidden, temperature):
            per_example_loss = simclr_contrastive_loss(hidden, hidden_norm=True, temperature=temperature)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

        # 训练步骤
        def train_step(inputs):
            with tf.GradientTape() as tape:
                x, aug_x, _ = inputs
                z1 = network(x)
                z2 = network(aug_x)
                hidden = tf.concat([z1, z2], axis=0)

                # 对比学习
                loss = compute_loss(hidden, temperature=temperature)

                # 权重衰减
                weight_decay = add_weight_decay(network, adjust_per_optimizer=False)
                loss += weight_decay

            weight_decay_metric.update_state(weight_decay)
            total_loss_metric.update_state(loss)

            # 梯度更新
            grads = tape.gradient(loss, network.trainable_variables)
            optimizer.apply_gradients(zip(grads, network.trainable_variables))
            return loss

        # `experimental_run_v2`将复制提供的计算并使用分布式输入运行它。
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.experimental_run_v2(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        for epoch in range(epochs):
            start = time.process_time()
            weight_decay_metric.reset_states()
            total_loss_metric.reset_states()

            step = 0
            for x in train_dist_dataset:

                # 分布式训练
                distributed_train_step(x)

                if step % 100 == 0:
                    template = 'Epoch {}, Step {}, Weight Decay Loss: {:.6f},  Total Loss: {:.6f}.'
                    weight_decay_loss = weight_decay_metric.result()
                    total_loss = total_loss_metric.result()
                    logging.info(template.format(epoch + 1,
                                                 str(step),
                                                 float(weight_decay_loss),
                                                 float(total_loss)))
                step += 1
            end = time.process_time()

            total_loss = total_loss_metric.result()
            template = 'Epoch {}, Loss: {:.6f}, Times: {:.6f}.'
            print(template.format(epoch + 1, float(total_loss), (end - start)))

            # 保存模型
            if save_model is True:
                network.save(model_file)
                print("Complete export saved models.")
