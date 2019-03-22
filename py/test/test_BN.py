import tensorflow as tf
import numpy as np
from os import path
import shutil
if path.exists("./log"):
    shutil.rmtree("./log")

x = tf.placeholder(tf.float32, [None, 100])
m, v = tf.nn.moments(x, [0])
tf.summary.histogram("x", x)
bn_x = tf.nn.batch_normalization(x, m, v, 0, 1, 0.01)
tf.summary.histogram("bn_x", bn_x)

w = tf.ones([100, 300])
b = tf.random.normal([1, 300])
pre = tf.matmul(bn_x, w) + b

tf.summary.histogram("pre", pre)


mean_, var_ = tf.nn.moments(pre, [0])

"""
scale * [(wx_plus_b-mean_)/sqrt(var+epsilon)]  +offset
"""
offset = tf.zeros([300])
scale = tf.ones([300])
epsilon = tf.ones([300])*0.1
bn_pre = tf.nn.batch_normalization(pre, mean_, var_, offset, scale, epsilon, 'batch_normalization')

tf.summary.histogram("bn_pre", bn_pre)
w2 = tf.random.normal([300, 20])
b2 = tf.random.normal([1, 20])
pre2 = tf.matmul(bn_pre, w2) + b2


tf.summary.histogram("pre2", pre2)
sess = tf.InteractiveSession()
merge_all = tf.summary.merge_all()
input_ = np.random.randint(0, 100, (1000,100))
input_t = tf.convert_to_tensor(input_,tf.float32)
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./log", sess.graph)
_, result = sess.run([merge_all, pre2], feed_dict={x: input_})
writer.add_summary(_)
print(result)
sess.close()