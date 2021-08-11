import tensorflow as tf
import math

from hiagi.self_supervised.optimizers import lars_optimizer


def build_optimizer(optimizer: str = 'adam',
                    momentum: float = 0.9,
                    weight_decay: float = 1e-6,
                    learning_rate: float = 10e-3):
    """
    返回优化器
    :param optimizer:
    :param momentum:
    :param weight_decay:
    :param learning_rate:
    :return:
    """
    if optimizer == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, momentum, nesterov=True)
    elif optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == 'lars':
        return lars_optimizer.LARSOptimizer(
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            exclude_from_weight_decay=[
                'batch_normalization', 'bias', 'head_supervised'
            ])
    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer))


def add_weight_decay(model: tf.keras.models.Model,
                     optimizer: str = 'adam',
                     adjust_per_optimizer: bool = True,
                     weight_decay: float = 1e-6):
    """
    计算权重衰减
    Compute weight decay from flags.
    :param model: 网络模型
    :param optimizer: 优化器
    :param adjust_per_optimizer:
    :param weight_decay: 权重衰减系数
    :return:
    """
    if adjust_per_optimizer and 'lars' in optimizer:
        # Weight decay are taking care of by optimizer for these cases.
        # Except for supervised head, which will be added here.
        l2_losses = [
            tf.nn.l2_loss(v)
            for v in model.trainable_variables
            if 'head_supervised' in v.name and 'bias' not in v.name
        ]
        if l2_losses:
            return weight_decay * tf.add_n(l2_losses)
        else:
            return 0

    # TODO(srbs): Think of a way to avoid name-based filtering here.
    # TODO: 需要补充其他Layer的名称
    l2_losses = [
        tf.nn.l2_loss(v)
        for v in model.trainable_weights
        if 'batch_normalization' not in v.name
    ]
    loss = weight_decay * tf.add_n(l2_losses)
    return loss


def get_train_steps(num_examples, train_steps=None, train_epochs: int = 100, train_batch_size: int = 32):
    """Determine the number of training steps."""
    return train_steps or (
            num_examples * train_epochs // train_batch_size + 1)


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    """
    def __init__(self, base_learning_rate, num_examples,
                 warmup_epochs: int = 10,
                 train_batch_size: int = 32,
                 train_steps: int = None,
                 train_epochs: int = 10,
                 learning_rate_scaling: str = 'linear',
                 name: str = None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name
        self.warmup_epochs = warmup_epochs
        self.train_batch_size = train_batch_size
        self.learning_rate_scaling = learning_rate_scaling
        self.train_steps = train_steps
        self.train_epochs = train_epochs

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(self.warmup_epochs * self.num_examples // self.train_batch_size))
            if self.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * self.train_batch_size / 256.
            elif self.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(self.train_batch_size)
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)

            # Cosine decay learning rate schedule
            total_steps = get_train_steps(self.num_examples)
            # TODO(srbs): Cache this object.
            cosine_decay = tf.keras.experimental.CosineDecay(scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'num_examples': self.num_examples,
            'warmup_epochs': self.warmup_epochs,
            'train_batch_size': self.train_batch_size,
            'learning_rate_scaling': self.learning_rate_scaling,
            'train_steps': self.train_steps,
            'train_epochs': self.train_epochs,
        }
