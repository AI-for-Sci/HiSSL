import numpy as np

import tensorflow as tf


class DINOLoss(tf.keras.layers.Layer):
    def __init__(self,
                 out_dim,
                 ncrops,
                 warmup_teacher_temp,
                 teacher_temp,
                 warmup_teacher_temp_epochs,
                 nepochs,
                 num_teachers=2,
                 student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()

        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.out_dim = out_dim
        self.num_teachers = num_teachers
        # self.register_buffer("center", tf.zeros([1, out_dim], dtype=tf.float32))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def build(self, input_shape):
        self.center = self.add_weight(name='center',
                                       shape=(1, self.out_dim),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # student_out = student_out.chunk(self.ncrops)
        student_out = tf.split(student_out, self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = tf.nn.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = tf.split(teacher_out, self.num_teachers)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue

                loss = tf.reduce_sum(-q * tf.nn.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += tf.reduce_mean(loss)
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    # @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = tf.reduce_sum(teacher_output, axis=0, keepdims=True)
        batch_center = tf.reduce_mean(batch_center)
        # dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output))

        batch_center = tf.cast(batch_center, dtype=tf.float32)

        # ema update
        new_center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.add_update((self.center, new_center), teacher_output)



if __name__ == '__main__':
    dino_loss = DINOLoss(
        out_dim=128,
        ncrops=5,
        warmup_teacher_temp=0.1,
        teacher_temp=0.01,
        warmup_teacher_temp_epochs=10,
        nepochs=100)

    teacher_out = np.random.randn(2, 128)
    student_out = np.random.randn(5, 128)

    loss = dino_loss(student_out, teacher_out, epoch=1)
    print("Loss: ", loss)
