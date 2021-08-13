# coding=utf-8
# Copyright 2020 The SimCLR Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""The main training pipeline."""

import json
import math
import os
import numpy as np
# import tensorflow.compat.v2 as tf
# import tensorflow_datasets as tfds

# 2021-08-11 只支持 tensorflow 2.1.0
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from tensorflow.keras import datasets

import sys

sys.path.append("../../")

from hissl.self_supervised.data import data as data_lib
from hissl.self_supervised.metrics import metrics
from hissl.self_supervised.simclr import model as model_lib
from hissl.self_supervised.losses import simclr_objective as obj_lib
from hissl.self_supervised.data.data import get_preprocess_fn

FLAGS = flags.FLAGS

# 2021-08-11 learning_rate 0.3 -> 0.0001
flags.DEFINE_float(
    'learning_rate', 0.001,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_enum(
    'learning_rate_scaling', 'linear', ['linear', 'sqrt'],
    'How to scale the learning rate as a function of batch size.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float('weight_decay', 1e-6, 'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm decay parameter.')

flags.DEFINE_integer(
    'train_batch_size', 32,
    'Batch size for training.')

flags.DEFINE_string(
    'train_split', 'train',
    'Split for training.')

flags.DEFINE_integer(
    'train_epochs', 10,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_steps', 0,
    'Number of steps to eval for. If not provided, evals over entire dataset.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_string(
    'eval_split', 'test',
    'Split for evaluation.')

# 2021-08-11 修改为 mnist
flags.DEFINE_string(
    'dataset', 'mnist',
    'Name of a dataset.')

flags.DEFINE_bool(
    'cache_dataset', False,
    'Whether to cache the entire dataset in memory. If the dataset is '
    'ImageNet, this is a very bad idea, but for smaller datasets it can '
    'improve performance.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_bool('lineareval_while_pretraining', False,
                  'Whether to finetune supervised head while pretraining.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for fine-tuning if a finetuning '
    'checkpoint does not already exist in model_dir.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string(
    'master', None,
    'Address/name of the TensorFlow master to use. By default, use an '
    'in-process master.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

# 2021-08-11 修改为 试用GPU 训练
flags.DEFINE_bool(
    'use_tpu', False,
    'Whether to run on TPU.')

flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')

flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

# 2021-08-11 修改为adam
flags.DEFINE_enum(
    'optimizer', 'adam', ['momentum', 'adam', 'lars'],
    'Optimizer to use.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_integer(
    'keep_hub_module_max', 1,
    'Maximum number of Hub modules to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means no projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', False,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

# 2021-08-11 resnet_depth 修改为18， 是最小的resnet
flags.DEFINE_integer(
    'resnet_depth', 18,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

# 2021-08-11, image_size修改为28，是mnist或者fashion-mnist的尺寸
flags.DEFINE_integer(
    'image_size', 28,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

# 简化代码
def get_salient_tensors_dict(include_projection_head):
    """Returns a dictionary of tensors."""
    graph = tf.compat.v1.get_default_graph()
    result = {}
    for i in range(1, 5):
        result['block_group%d' % i] = graph.get_tensor_by_name(
            'resnet/block_group%d/block_group%d:0' % (i, i))
    result['initial_conv'] = graph.get_tensor_by_name(
        'resnet/initial_conv/Identity:0')
    result['initial_max_pool'] = graph.get_tensor_by_name(
        'resnet/initial_max_pool/Identity:0')
    result['final_avg_pool'] = graph.get_tensor_by_name('resnet/final_avg_pool:0')
    result['logits_sup'] = graph.get_tensor_by_name(
        'head_supervised/logits_sup:0')
    if include_projection_head:
        result['proj_head_input'] = graph.get_tensor_by_name(
            'projection_head/proj_head_input:0')
        result['proj_head_output'] = graph.get_tensor_by_name(
            'projection_head/proj_head_output:0')
    return result


def build_saved_model(model, include_projection_head=True):
    """Returns a tf.Module for saving to SavedModel."""

    class SimCLRModel(tf.Module):
        """Saved model for exporting to hub."""

        def __init__(self, model):
            self.model = model
            # This can't be called `trainable_variables` because `tf.Module` has
            # a getter with the same name.
            self.trainable_variables_list = model.trainable_variables

        @tf.function
        def __call__(self, inputs, trainable):
            self.model(inputs, training=trainable)
            return get_salient_tensors_dict(include_projection_head)

    module = SimCLRModel(model)
    input_spec = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)
    module.__call__.get_concrete_function(input_spec, trainable=True)
    module.__call__.get_concrete_function(input_spec, trainable=False)
    return module


def save(model, global_step):
    """Export as SavedModel for finetuning and inference."""
    saved_model = build_saved_model(model)
    export_dir = os.path.join(FLAGS.model_dir, 'saved_model')
    checkpoint_export_dir = os.path.join(export_dir, str(global_step))
    if tf.io.gfile.exists(checkpoint_export_dir):
        tf.io.gfile.rmtree(checkpoint_export_dir)
    tf.saved_model.save(saved_model, checkpoint_export_dir)

    if FLAGS.keep_hub_module_max > 0:
        # Delete old exported SavedModels.
        exported_steps = []
        for subdir in tf.io.gfile.listdir(export_dir):
            if not subdir.isdigit():
                continue
            exported_steps.append(int(subdir))
        exported_steps.sort()
        for step_to_delete in exported_steps[:-FLAGS.keep_hub_module_max]:
            tf.io.gfile.rmtree(os.path.join(export_dir, str(step_to_delete)))


def try_restore_from_checkpoint(model, global_step, optimizer):
    """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
    checkpoint = tf.train.Checkpoint(
        model=model, global_step=global_step, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=FLAGS.model_dir,
        max_to_keep=FLAGS.keep_checkpoint_max)
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        # Restore model weights, global step, optimizer states
        logging.info('Restoring from latest checkpoint: %s', latest_ckpt)
        # 2021-08-11 修改checkpoint为_checkpoint
        checkpoint_manager._checkpoint.restore(latest_ckpt).expect_partial()
    elif FLAGS.checkpoint:
        # Restore model weights only, but not global step and optimizer states
        logging.info('Restoring from given checkpoint: %s', FLAGS.checkpoint)
        checkpoint_manager2 = tf.train.CheckpointManager(
            tf.train.Checkpoint(model=model),
            directory=FLAGS.model_dir,
            max_to_keep=FLAGS.keep_checkpoint_max)
        checkpoint_manager2._checkpoint.restore(FLAGS.checkpoint).expect_partial()
        if FLAGS.zero_init_logits_layer:
            model = checkpoint_manager2._checkpoint.model
            output_layer_parameters = model.supervised_head.trainable_weights
            logging.info('Initializing output layer parameters %s to zero',
                         [x.op.name for x in output_layer_parameters])
            for x in output_layer_parameters:
                x.assign(tf.zeros_like(x))

    return checkpoint_manager


def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except TypeError:
        return False


def perform_evaluation(model, builder, eval_steps, ckpt, strategy, topology, ds=None):
    """Perform evaluation."""
    if FLAGS.train_mode == 'pretrain' and not FLAGS.lineareval_while_pretraining:
        logging.info('Skipping eval during pretraining without linear eval.')
        return
    # Build input pipeline.
    if ds is None:
        ds = build_distributed_dataset(builder, FLAGS.eval_batch_size, False,
                                                strategy, topology)
    summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

    # Build metrics.
    with strategy.scope():
        regularization_loss = tf.keras.metrics.Mean('eval/regularization_loss')
        label_top_1_accuracy = tf.keras.metrics.Accuracy(
            'eval/label_top_1_accuracy')
        label_top_5_accuracy = tf.keras.metrics.TopKCategoricalAccuracy(
            5, 'eval/label_top_5_accuracy')
        all_metrics = [
            regularization_loss, label_top_1_accuracy, label_top_5_accuracy
        ]

        # Restore checkpoint.
        logging.info('Restoring from %s', ckpt)
        checkpoint = tf.train.Checkpoint(
            model=model, global_step=tf.Variable(0, dtype=tf.int64))
        checkpoint.restore(ckpt).expect_partial()
        global_step = checkpoint.global_step
        logging.info('Performing eval at step %d', global_step.numpy())

    def single_step(features, labels):
        _, supervised_head_outputs = model(features, training=False)
        assert supervised_head_outputs is not None
        outputs = supervised_head_outputs
        l = labels['labels']
        metrics.update_finetune_metrics_eval(label_top_1_accuracy,
                                             label_top_5_accuracy, outputs, l)
        reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
        regularization_loss.update_state(reg_loss)

    with strategy.scope():

        @tf.function
        def run_single_step(iterator):
            images, labels = next(iterator)
            features, labels = images, {'labels': labels}
            strategy.run(single_step, (features, labels))

        iterator = iter(ds)
        for i in range(eval_steps):
            run_single_step(iterator)
            logging.info('Completed eval for %d / %d steps', i + 1, eval_steps)
        logging.info('Finished eval for %s', ckpt)

    # Write summaries
    cur_step = global_step.numpy()
    logging.info('Writing summaries for %d step', cur_step)
    with summary_writer.as_default():
        metrics.log_and_write_metrics_to_summary(all_metrics, cur_step)
        summary_writer.flush()

    # Record results as JSON.
    result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
    result = {metric.name: metric.result().numpy() for metric in all_metrics}
    result['global_step'] = global_step.numpy()
    logging.info(result)
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    result_json_path = os.path.join(
        FLAGS.model_dir, 'result_%d.json' % result['global_step'])
    with tf.io.gfile.GFile(result_json_path, 'w') as f:
        json.dump({k: float(v) for k, v in result.items()}, f)
    flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
    with tf.io.gfile.GFile(flag_json_path, 'w') as f:
        serializable_flags = {}
        for key, val in FLAGS.flag_values_dict().items():
            # Some flag value types e.g. datetime.timedelta are not json serializable,
            # filter those out.
            if json_serializable(val):
                serializable_flags[key] = val
        json.dump(serializable_flags, f)

    # Export as SavedModel for finetuning and inference.
    save(model, global_step=result['global_step'])

    return result


def _restore_latest_or_from_pretrain(checkpoint_manager):
    """Restores the latest ckpt if training already.
    Or restores from FLAGS.checkpoint if in finetune mode.
    Args:
      checkpoint_manager: tf.traiin.CheckpointManager.
    """
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        # The model is not build yet so some variables may not be available in
        # the object graph. Those are lazily initialized. To suppress the warning
        # in that case we specify `expect_partial`.
        logging.info('Restoring from %s', latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
    elif FLAGS.train_mode == 'finetune':
        # Restore from pretrain checkpoint.
        assert FLAGS.checkpoint, 'Missing pretrain checkpoint.'
        logging.info('Restoring from %s', FLAGS.checkpoint)
        checkpoint_manager.checkpoint.restore(FLAGS.checkpoint).expect_partial()
        # TODO(iamtingchen): Can we instead use a zeros initializer for the
        # supervised head?
        if FLAGS.zero_init_logits_layer:
            model = checkpoint_manager.checkpoint.model
            output_layer_parameters = model.supervised_head.trainable_weights
            logging.info('Initializing output layer parameters %s to zero',
                         [x.op.name for x in output_layer_parameters])
            for x in output_layer_parameters:
                x.assign(tf.zeros_like(x))


def build_input_fn(x_train, y_train, global_batch_size, topology, is_training, strategy, num_classes=10):
    """Build input function.
    Args:
      builder: TFDS builder for specified dataset.
      global_batch_size: Global batch size.
      topology: An instance of `tf.tpu.experimental.Topology` or None.
      is_training: Whether to build in training mode.
    Returns:
      A function that accepts a dict of params and returns a tuple of images and
      features, to be used as the input_fn in TPUEstimator.
    """

    # def _input_fn(input_context):
    """Inner input function."""
    batch_size = global_batch_size // strategy.num_replicas_in_sync  #input_context.get_per_replica_batch_size(global_batch_size)
    logging.info('Global batch size: %d', global_batch_size)
    logging.info('Per-replica batch size: %d', batch_size)
    preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    # num_classes = np.unique(y_train)  # builder.info.features['label'].num_classes

    def map_fn(image, label):
        """Produces multiple transformations of the same batch."""
        if is_training and FLAGS.train_mode == 'pretrain':
            xs = []
            for _ in range(2):  # Two transformations
                xs.append(preprocess_fn_pretrain(image))
            image = tf.concat(xs, -1)
        else:
            image = preprocess_fn_finetune(image)
        label = tf.one_hot(label, num_classes)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if FLAGS.cache_dataset:
        dataset = dataset.cache()
    if is_training:
        options = tf.data.Options()
        options.experimental_deterministic = False
        options.experimental_slack = True
        dataset = dataset.with_options(options)
        buffer_multiplier = 50 if FLAGS.image_size <= 32 else 10
        dataset = dataset.shuffle(batch_size * buffer_multiplier)
        # dataset = dataset.repeat(-1)
    dataset = dataset.map(
        map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

    # return _input_fn


def build_distributed_dataset(x_train, y_train, batch_size, is_training, strategy,
                              topology):
    input_fn = build_input_fn(x_train, y_train, batch_size, topology, is_training, strategy)

    # 2021-08-11 Issue for tensorflow 2.1.0
    # AttributeError: 'MirroredStrategy' object has no attribute 'distribute_datasets_from_function'
    # return strategy.distribute_datasets_from_function(input_fn)
    # print("input_fn: ", input_fn)
    return input_fn #strategy.experimental_distribute_dataset(input_fn)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # 数据比较难下载，改为
    # builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
    # builder.download_and_prepare()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 简化数据输入，只测试 mnist 和 fashion-mnist
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # reshape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # convert from int to float
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # rescale values
    x_train /= 255.0
    x_test /= 255.0

    # print("x_train: ", x_train[0])

    x_train = tf.concat([x_train, x_train, x_train], axis=-1)
    print("image: ", x_train.shape)
    x_test = tf.concat([x_test, x_test, x_test], axis=-1)
    print("image: ", x_test.shape)


    num_train_examples = len(y_train)  # builder.info.splits[FLAGS.train_split].num_examples
    num_eval_examples = len(y_test)  # builder.info.splits[FLAGS.eval_split].num_examples
    num_classes = np.unique(y_train)  # .builder.info.features['label'].num_classes

    train_steps = model_lib.get_train_steps(num_train_examples)
    eval_steps = FLAGS.eval_steps or int(
        math.ceil(num_eval_examples / FLAGS.eval_batch_size))
    epoch_steps = int(round(num_train_examples / FLAGS.train_batch_size))

    logging.info('# train examples: %d', num_train_examples)
    logging.info('# train_steps: %d', train_steps)
    logging.info('# eval examples: %d', num_eval_examples)
    logging.info('# eval steps: %d', eval_steps)

    checkpoint_steps = (
            FLAGS.checkpoint_steps or (FLAGS.checkpoint_epochs * epoch_steps))

    logging.info('# checkpoint_steps: %d', checkpoint_steps)

    topology = None
    # 简化代码
    # For (multiple) GPUs.
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Running using MirroredStrategy on %d replicas',
                 strategy.num_replicas_in_sync)

    # 简化代码，不支持多GPU，减少痛苦
    # with strategy.scope():
    #     model = model_lib.Model(num_classes)
    model = model_lib.Model(num_classes, model_name='lenet')

    if FLAGS.mode == 'eval':
        # for ckpt in tf.train.checkpoints_iterator(
        #         FLAGS.model_dir, min_interval_secs=15):
        #     ds = build_distributed_dataset(x_test, y_test,
        #                                    FLAGS.eval_batch_size,
        #                                    True,
        #                                    strategy,
        #                                    topology)
        #     result = perform_evaluation(model, None, eval_steps, ckpt, strategy,
        #                                 topology, ds=ds)
        #     if result['global_step'] >= train_steps:
        #         logging.info('Eval complete. Exiting...')
        #         return
        pass
    else:
        summary_writer = tf.summary.create_file_writer(FLAGS.model_dir)

        # 简化，不使用多GPU
        # with strategy.scope():
        # Build input pipeline.
        ds = build_distributed_dataset(x_train, y_train,
                                       FLAGS.train_batch_size,
                                       True,
                                       strategy,
                                       topology)

        # Build LR schedule and optimizer.
        learning_rate = model_lib.WarmUpAndCosineDecay(FLAGS.learning_rate,
                                                       num_train_examples)
        optimizer = model_lib.build_optimizer(learning_rate)

        # Build metrics.
        all_metrics = []  # For summaries.
        weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
        total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
        all_metrics.extend([weight_decay_metric, total_loss_metric])
        if FLAGS.train_mode == 'pretrain':
            contrast_loss_metric = tf.keras.metrics.Mean('train/contrast_loss')
            contrast_acc_metric = tf.keras.metrics.Mean('train/contrast_acc')
            contrast_entropy_metric = tf.keras.metrics.Mean(
                'train/contrast_entropy')
            all_metrics.extend([
                contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric
            ])
        if FLAGS.train_mode == 'finetune' or FLAGS.lineareval_while_pretraining:
            supervised_loss_metric = tf.keras.metrics.Mean('train/supervised_loss')
            supervised_acc_metric = tf.keras.metrics.Mean('train/supervised_acc')
            all_metrics.extend([supervised_loss_metric, supervised_acc_metric])

        # Restore checkpoint if available.
        checkpoint_manager = try_restore_from_checkpoint(model, optimizer.iterations, optimizer)

        steps_per_loop = checkpoint_steps
        # with strategy.scope():
        for epoch in range(FLAGS.train_epochs):
            print("Epoch: ", epoch, '=============================================================')
            weight_decay_metric.reset_states()
            total_loss_metric.reset_states()
            if FLAGS.train_mode == 'pretrain':
                contrast_loss_metric.reset_states()
                contrast_acc_metric.reset_states()
                contrast_entropy_metric.reset_states()


            for step, (features, labels) in enumerate(ds):
                with tf.GradientTape() as tape:
                    # print("Feature: " , features.shape)
                    projection_head_outputs, supervised_head_outputs, hidden = model(
                        features, training=True)
                    loss = None
                    if projection_head_outputs is not None:
                        outputs = projection_head_outputs
                        con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                            outputs,
                            hidden_norm=FLAGS.hidden_norm,
                            temperature=FLAGS.temperature,
                            strategy=strategy)
                        if loss is None:
                            loss = con_loss
                        else:
                            loss += con_loss
                        metrics.update_pretrain_metrics_train(contrast_loss_metric,
                                                              contrast_acc_metric,
                                                              contrast_entropy_metric,
                                                              con_loss, logits_con,
                                                              labels_con)
                    # if supervised_head_outputs is not None:
                    #     outputs = supervised_head_outputs
                    #     l = labels['labels']
                    #     if FLAGS.train_mode == 'pretrain' and FLAGS.lineareval_while_pretraining:
                    #         l = tf.concat([l, l], 0)
                    #     sup_loss = obj_lib.add_supervised_loss(labels=l, logits=outputs)
                    #     if loss is None:
                    #         loss = sup_loss
                    #     else:
                    #         loss += sup_loss
                    #     metrics.update_finetune_metrics_train(supervised_loss_metric,
                    #                                           supervised_acc_metric, sup_loss,
                    #                                           l, outputs)

                    weight_decay = model_lib.add_weight_decay(
                        model, adjust_per_optimizer=True)
                    weight_decay_metric.update_state(weight_decay)
                    loss += weight_decay
                    total_loss_metric.update_state(loss)
                    # The default behavior of `apply_gradients` is to sum gradients from all
                    # replicas so we divide the loss by the number of replicas so that the
                    # mean gradient is applied.
                    loss = loss / strategy.num_replicas_in_sync
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 50 == 0:
                    template = 'Epoch {}, Step {}, train Loss: {:0.5f}, weight decay: {:0.5f}. '
                    print(template.format(epoch + 1,
                                          str(step),
                                          total_loss_metric.result(),
                                          weight_decay_metric.result()
                                          ))
            valid_ds = build_distributed_dataset(x_test, y_test,
                                                 FLAGS.train_batch_size,
                                                 False,
                                                 strategy,
                                                 topology)
            embedding_outputs = np.zeros((int(num_eval_examples * 1.2), FLAGS.proj_out_dim * 4), dtype=np.float32)
            embedding_labels = np.zeros((int(num_eval_examples * 1.2)), dtype=np.int32)
            valid_samples = 0
            for step, (features, labels) in enumerate(valid_ds):
                with tf.GradientTape() as tape:
                    # print("Feature: " , features.shape)
                    projection_head_outputs, supervised_head_outputs, hidden = model(
                        features, training=True)
                    if hidden is not None:
                        # 记录特征向量和标签
                        z1 = hidden
                        z1 = tf.math.l2_normalize(z1, axis=-1)
                        embedding_outputs[valid_samples: valid_samples + len(features)] = z1
                        embedding_labels[valid_samples: valid_samples + len(features)] = tf.argmax(labels, axis=-1)
                        valid_samples += len(features)

            # Calls to tf.summary.xyz lookup the summary writer resource which is
            # set by the summary writer's context manager.
            with summary_writer.as_default():
                checkpoint_manager.save(epoch)
                logging.info('Completed: %d / %d steps', epoch, train_steps)
                metrics.log_and_write_metrics_to_summary(all_metrics, epoch)
                tf.summary.scalar(
                    'learning_rate',
                    learning_rate(tf.cast(epoch, dtype=tf.float32)),
                    epoch)
                summary_writer.flush()
            for metric in all_metrics:
                metric.reset_states()
            logging.info('Training complete...')

            import matplotlib.pyplot as plt
            embedding_outputs = embedding_outputs[:valid_samples]
            embedding_labels = embedding_labels[:valid_samples]

            if FLAGS.train_mode == 'pretrain':
                import umap
                fit = umap.UMAP()
                umap = fit.fit_transform(embedding_outputs)

                def pretrain_embedding_plot():
                    plt.clf()
                    fig = plt.figure(figsize=(15, 10))
                    fig.subplots_adjust(hspace=0.1, wspace=0.1)
                    for ii in range(1, 2):
                        ax = fig.add_subplot(1, 1, ii)
                        plt.sca(ax)
                        if ii == 1:
                            num_labels = len(np.unique(embedding_labels))
                            if num_labels == 1:
                                num_labels = 10
                            print("num_labels: ", num_labels)
                            plt.scatter(umap[:, 0], umap[:, 1], s=2, alpha=0.75, c=embedding_labels, cmap='Spectral')
                            plt.gca().set_aspect('equal', 'datalim')
                            plt.colorbar(boundaries=np.arange(num_labels + 1) - 0.5).set_ticklabels(
                                np.array(range(num_labels)))
                            plt.title('Visualizing label through UMAP', fontsize=16);
                        else:
                            break

                    save_path = './figures'
                    if os.path.exists(save_path) is False:
                        os.mkdir(save_path)
                    plt.savefig(os.path.join(save_path,'Visualizing label through UMAP_Epoch_{}_SimCLR.png'.format(epoch + 1)))

                # 打印向量Embedding
                pretrain_embedding_plot()

        # if FLAGS.mode == 'train_then_eval':
        #     ds = build_distributed_dataset(x_test, y_test,
        #                                    FLAGS.eval_batch_size,
        #                                    True,
        #                                    strategy,
        #                                    topology)
        #     perform_evaluation(model, None, eval_steps,
        #                        checkpoint_manager.latest_checkpoint, strategy,
        #                        topology, ds=ds)


if __name__ == '__main__':
    # tf.compat.v1.enable_v2_behavior()
    # For outside compilation of summaries on TPU.
    # tf.config.set_soft_device_placement(True)
    app.run(main)
