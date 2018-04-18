'''
    TensorFlow系列教程(2)——手写数字的识别
    https://www.youtube.com/watch?v=gx7iEa9Q-Vs
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Read data
# one-hot vector is a vector which is 0 in most dimensions, and 1 in a single dimension
# For example, 3 would be [0,0,0,1,0,0,0,0,0,0]
mnist = input_data.read_data_sets('./data/MNIST', one_hot=True)

x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
# mnist.train.labels is a [55000, 10] array of floats
y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

batch_size = 1000


def add_layer(input_data, input_num, output_num, activation_fun=None):
    w = tf.Variable(initial_value=tf.random_normal(shape=[input_num, output_num]))
    b = tf.Variable(initial_value=tf.random_normal(shape=[1, output_num]))
    # output = input_data * weight + bias
    output = tf.add(tf.matmul(input_data, w), b)

    if activation_fun:
        output = activation_fun(output)
    return output


def build_nn(data):
    hidden_layer1 = add_layer(data, 784, 100, activation_fun=tf.nn.sigmoid)
    hidden_layer2 = add_layer(hidden_layer1, 100, 50, activation_fun=tf.nn.sigmoid)
    output_layer = add_layer(hidden_layer2, 50, 10)
    return output_layer


def train_nn(data):
    # output of NN
    output = build_nn(data)

    # softmax used for vector compairation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    
    #
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            epoch_cost = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                x_data, y_data = mnist.train.next_batch(batch_size)
                cost, _ = sess.run([loss, optimizer], feed_dict={x: x_data, y: y_data})
                epoch_cost += cost
            print('Epoch', i, ": ", epoch_cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(output, 1)), tf.float32))
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("accuracy: ", acc)


train_nn(x)
