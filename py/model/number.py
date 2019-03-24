from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append(r"C:\Users\wangheng\workspace\CLionWorkSpace\clionOpencv\py\utils")
import cv2
import numpy as np
from py.utils import nn_utils
import tensorflow as tf


def log(value):
    print(value)


class NumModel:
    def __init__(self):
        self.df = pd.read_csv(r"C:\Users\wangheng\Documents\software_cup\dataframe1.csv", index_col=0)
        values = self.df.values
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(values[:, 0], values[:, 2:],
                                                                                test_size=0.3, random_state=0,
                                                                                shuffle=True)


class batch:
    def __init__(self, batch_size=128):
        self.start = 0
        self.offset = batch_size
        self.has_next_batch = True
        model = NumModel()
        self.train_x, self.train_y = model.train_x, model.train_y
        self.end = self.train_x.shape[0]

    def get_next_batch(self):
        global one_batch_path_x, one_batch_y
        if self.start + self.offset <= self.end:
            one_batch_path_x = self.train_x[self.start: self.start + self.offset]
            one_batch_y = self.train_y[self.start: self.start + self.offset]
            self.start += self.offset
            if self.start >= self.end:
                self.has_next_batch = False
            else:
                self.has_next_batch = True
        elif self.start + self.offset > self.end:
            one_batch_path_x = self.train_x[self.start:]
            one_batch_y = self.train_y[self.start:]
            self.start = self.end + 1
            self.has_next_batch = False
        train_x = []
        for f in one_batch_path_x:
            imread = cv2.imread(f)
            train_x.append(imread)
        train_x = np.asarray(train_x, np.float)
        train_y = np.asarray(one_batch_y, np.float)
        return train_x, train_y


class model:
    def __init__(self):
        self.config = tf.ConfigProto(allow_soft_placement=True)

    def build(self):
        with tf.name_scope("input"):
            X = tf.placeholder(tf.float32, [None, 41, 28, 3], "input_x")
            y = tf.placeholder(tf.float32, [None, 11], "input_y")

        conv1 = nn_utils.conv(X, [3, 3, 3, 128], [1, 1], True, "leaky_relu", device="/GPU:0")
        max1 = nn_utils.max_pooling(conv1, [2, 2], [2, 2])
        conv2 = nn_utils.conv(max1, [3, 3, 128, 512], [1, 1], True, "leaky_relu")
        max2 = nn_utils.max_pooling(conv2, [2, 2], [2, 2])
        conv3 = nn_utils.conv(max2, [3, 3, 512, 1024], [1, 1], True, "leaky_relu")
        conv4 = nn_utils.conv(conv3, [3, 3, 1024, 1024], [1, 1], True, "leaky_relu")
        conv5 = nn_utils.conv(conv4, [3, 3, 1024, 512], [1, 1], True, "leaky_relu")
        max3 = nn_utils.max_pooling(conv5, [2, 2], [2, 2])
        flat_max3 = tf.reshape(max3, (-1, max3.shape[1].value*max3.shape[2].value*max3.shape[3].value))
        fc1 = nn_utils.fc(flat_max3, 1024, keep_prob=0.7)
        fc2 = nn_utils.fc(fc1, 512, "relu", keep_prob=0.7)
        fc3 = nn_utils.fc(fc2, 128, "relu", keep_prob=0.7)
        fc4 = nn_utils.fc(fc3, 11)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=fc4, labels=y)
        loss = tf.reduce_mean(loss)
        tf.summary.scalar("loss", loss)
        opt = tf.train.AdamOptimizer().minimize(loss)

        log("conv1:\t" + "input-->" + str(X.shape.as_list()[1:]) + "\toutput-->" + str(conv1.shape.as_list()[1:]))
        log("max1:\t" + "input-->" + str(conv1.shape.as_list()[1:]) + "\toutput-->" + str(max1.shape.as_list()[1:]))
        log("conv2:\t" + "input-->" + str(max1.shape.as_list()[1:]) + "\toutput-->" + str(conv2.shape.as_list()[1:]))
        log("max2:\t" + "input-->" + str(conv2.shape.as_list()[1:]) + "\toutput-->" + str(max2.shape.as_list()[1:]))
        log("conv3:\t" + "input-->" + str(max2.shape.as_list()[1:]) + "\toutput-->" + str(conv3.shape.as_list()[1:]))
        log("conv4:\t" + "input-->" + str(conv3.shape.as_list()[1:]) + "\toutput-->" + str(conv4.shape.as_list()[1:]))
        log("conv5:\t" + "input-->" + str(conv4.shape.as_list()[1:]) + "\toutput-->" + str(conv5.shape.as_list()[1:]))
        log("max3:\t" + "input-->" + str(conv5.shape.as_list()[1:]) + "\toutput-->" + str(max3.shape.as_list()[1:]))

        sess = tf.InteractiveSession(config=self.config)
        writer = tf.summary.FileWriter("./log", sess.graph)
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1000):
            mybatch = batch()
            x_ = None
            y_ = None
            while mybatch.has_next_batch:
                x_, y_ = mybatch.get_next_batch()
                print(x_.shape, y_.shape)
                sess.run(opt, feed_dict={X: x_, y: y_})
            if i %10 == 0:
                _, loss_ = sess.run([merged, loss], feed_dict={X:x_, y:y_})
                writer.add_summary(_, i)
                print("step %d, loss value is: %.4f"%(i, loss_))
            if i % 100 ==0:
                saver.save(sess, "./mymodel/number.ckpt")
        sess.close()


if __name__ == '__main__':
    # mybatch = batch(1000)
    # while mybatch.has_next_batch:
    #     trainx, trainy = mybatch.get_next_batch()
    #     print(trainx.shape, trainy.shape)
    mymodel = model()
    mymodel.build()
