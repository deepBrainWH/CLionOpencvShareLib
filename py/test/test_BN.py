import tensorflow as tf
import numpy as np
from os import path
import shutil
if path.exists("./log"):
    shutil.rmtree("./log")

input_ = np.random.randint(0, 100, (1000,100))
output_ = np.random.randint(0, 100, (1000, 10))


x = tf.placeholder(tf.float32, [None, 100])
y = tf.placeholder(tf.float32, [None, 10])
m, v = tf.nn.moments(x, [0])
tf.summary.histogram("x", x)
bn_x = tf.nn.batch_normalization(x, m, v, 0, 1, 0.01)
tf.summary.histogram("bn_x", bn_x)

w = tf.Variable(tf.ones([100, 300]))
b = tf.Variable(tf.random.normal([1, 300]))
pre = tf.matmul(bn_x, w) + b
pre = tf.nn.sigmoid(pre)
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
w2 = tf.Variable(tf.random.normal([300, 10]))
b2 = tf.Variable(tf.random.normal([1, 10]))
pre2 = tf.matmul(bn_pre, w2) + b2
pre2 = tf.nn.relu(pre2)
tf.summary.histogram("pre2", pre2)

loss = tf.losses.mean_squared_error(y, pre2)
tf.summary.scalar("loss", loss)
opt = tf.train.AdadeltaOptimizer(0.001).minimize(tf.reduce_sum(loss))

sess = tf.InteractiveSession()
merge_all = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter("./log", sess.graph)
for i in range(200000):
    result = sess.run([opt], feed_dict={x: input_,  y: output_})
    if i % 10 == 0:
        _, los_ = sess.run([merge_all, loss], feed_dict={x: input_,  y: output_})
        print("step %d, loss value is %.4f" % (i, los_))
        writer.add_summary(_, i)
writer.close()
sess.close()