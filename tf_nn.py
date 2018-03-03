import numpy as np
import tensorflow as tf

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None,n_features))   # do not determain the rows

W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x, W)
z = tf.add(xW, b)

# activation funtion, rectified linear unit
# a = tf.nn.relu()
a = tf.sigmoid(z)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x:np.random.random([1, n_features])})

print(layer_out)