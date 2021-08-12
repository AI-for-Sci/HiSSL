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
"""Contains definitions for the post-activation form of Residual Networks.
Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from absl import flags
# import tensorflow.compat.v2 as tf
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, add, Embedding, multiply, subtract, add, dot, Dot
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5

class LeNet(tf.keras.layers.Layer):
    def __init__(self,
                 data_format='channels_last',
                 dropblock_keep_probs=0.95,
                 hidden_size=512,
                 **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.data_format = data_format

        self.con2d_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')
        self.con2d_2 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.maxpooling2d_1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout_1 = Dropout(rate=1 - dropblock_keep_probs)

        self.con2d_3 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.con2d_4 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.maxpooling2d_2 = MaxPooling2D(pool_size=(2, 2))

        # x = Flatten()(x)
        self.fc_1 = Dense(hidden_size, activation='relu', name='project_1')
        self.bn_1 = BatchNormalization()
        self.drop_1 = Dropout(rate=1 - dropblock_keep_probs)
        self.fc_2 = Dense(hidden_size, activation='relu', name='project_2')
        self.bn_2 = BatchNormalization()
        self.drop_2 = Dropout(rate=1 - dropblock_keep_probs)
        self.fc_3 = Dense(hidden_size, activation='relu', name='project_3')

    def call(self, inputs, training):
        x = self.con2d_1(inputs)
        x = self.con2d_2(x)
        x = self.maxpooling2d_1(x)
        x = self.dropout_1(x)

        x = self.con2d_3(x)
        x = self.con2d_4(x)
        x = self.maxpooling2d_2(x)

        x = Flatten()(x)
        x = self.fc_1(x)
        # x = self.bn_1(x)
        # x = self.drop_1(x)
        # x = self.fc_2(x)
        # x = self.bn_2(x)
        # x = self.drop_2(x)
        # x = self.fc_3(x)
        return x

