import tensorflow as tf
from py.utils import nn_utils


class myyolo:
    def __init__(self):
        self.classes = ['card', 'number']
        self.colors = {'card': (200, 0, 0), 'number': (0, 200, 0)}
        self.image = tf.placeholder(tf.float32, [None, 390, 520, 3], 'input')

    def model(self):
        tf.nn.batch_normalization(self.image,)
        conv1 = nn_utils.conv(self.image, [3, 3, 3, 16], [1, 1],
                              'leaky_relu')  # 195*260*3
        max1 = nn_utils.max_pooling(conv1, [2, 2], [2, 2])

        conv12 = nn_utils.conv(max1, [3,3,32, 32], [2,2], )
