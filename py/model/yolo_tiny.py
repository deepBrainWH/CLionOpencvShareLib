import tensorflow as tf
from py.utils import nn_utils


def log(value):
    print(value)


class myyolo:
    def __init__(self):
        self.classes = ['card', 'number']
        self.colors = {'card': (200, 0, 0), 'number': (0, 200, 0)}
        self.image = tf.placeholder(tf.float32, [None, 390, 520, 3], 'input')

    def model(self):
        conv1 = nn_utils.conv(self.image, [3, 3, 3, 16], [1, 1], True, 'leaky_relu')  # 390*520*32
        max1 = nn_utils.max_pooling(conv1, [2, 2], [2, 2])  # 195*260*32
        conv2 = nn_utils.conv(max1, [3, 3, 16, 32], [1, 1], True, 'leaky_relu')
        max2 = nn_utils.max_pooling(conv2, [2, 2], [2, 2])
        conv3 = nn_utils.conv(max2, [3, 3, 32, 64], [1, 1], True, "leaky_relu")
        max3 = nn_utils.max_pooling(conv3, [2, 2], [2, 2])
        conv4 = nn_utils.conv(max3, [3, 3, 64, 128], [1, 1], True, "leaky_relu")
        max4 = nn_utils.max_pooling(conv4, [2, 2], [2, 2])
        conv5 = nn_utils.conv(max4, [3, 3, 128, 256], [1, 1], True, "leaky_relu")
        max5 = nn_utils.max_pooling(conv5, [2, 2], [2, 2])
        conv6 = nn_utils.conv(max5, [3, 3, 256, 512], [1, 1], True, "leaky_relu")
        max6 = nn_utils.max_pooling(conv6, [2, 2], [1, 1])
        conv7 = nn_utils.conv(max6, [3, 3, 512, 1024], [1, 1], True, "leaky_relu")
        conv8 = nn_utils.conv(conv7, [3, 3, 1024, 512], [1, 1], True, "leaky_relu")
        conv9 = nn_utils.conv(conv8, [1,1,512, 425],[1,1], True, "leaky_relu")


        log("conv1:\t" + "input-->" + str(self.image.shape.as_list()[1:]) + "\toutput-->" + str(
            conv1.shape.as_list()[1:]))
        log("max1:\t" + "input-->" + str(conv1.shape.as_list()[1:]) + "\toutput-->" + str(max1.shape.as_list()[1:]))
        log("conv2:\t" + "input-->" + str(max1.shape.as_list()[1:]) + "\toutput-->" + str(conv2.shape.as_list()[1:]))
        log("max2:\t" + "input-->" + str(conv2.shape.as_list()[1:]) + "\toutput-->" + str(max2.shape.as_list()[1:]))
        log("conv3:\t" + "input-->" + str(max2.shape.as_list()[1:]) + "\toutput-->" + str(conv3.shape.as_list()[1:]))
        log("max3:\t" + "input-->" + str(conv3.shape.as_list()[1:]) + "\toutput-->" + str(max3.shape.as_list()[1:]))
        log("conv4:\t" + "input-->" + str(max3.shape.as_list()[1:]) + "\toutput-->" + str(conv4.shape.as_list()[1:]))
        log("max4:\t" + "input-->" + str(conv4.shape.as_list()[1:]) + "\toutput-->" + str(max4.shape.as_list()[1:]))
        log("conv5:\t" + "input-->" + str(max4.shape.as_list()[1:]) + "\toutput-->" + str(conv5.shape.as_list()[1:]))
        log("max5:\t" + "input-->" + str(conv5.shape.as_list()[1:]) + "\toutput-->" + str(max5.shape.as_list()[1:]))
        log("conv6:\t" + "input-->" + str(max5.shape.as_list()[1:]) + "\toutput-->" + str(conv6.shape.as_list()[1:]))
        log("max6:\t" + "input-->" + str(conv6.shape.as_list()[1:]) + "\toutput-->" + str(max6.shape.as_list()[1:]))
        log("conv7:\t" + "input-->" + str(conv7.shape.as_list()[1:]) + "\toutput-->" + str(conv7.shape.as_list()[1:]))
        log("conv8:\t" + "input-->" + str(conv8.shape.as_list()[1:]) + "\toutput-->" + str(conv8.shape.as_list()[1:]))
        log("conv9:\t"+"input-->"+str(conv9.shape.as_list()[1:])+"\toutput-->"+str(conv9.shape.as_list()[1:]))





if __name__ == '__main__':
    my = myyolo()
    my.model()
