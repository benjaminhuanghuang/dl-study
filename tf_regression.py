import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))
# target m=0.5 b=5
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

# concatemenaste data
my_data = pd.concat([x_df, y_df], axis=1)
my_data.sample(n=250).plot(kind='scatter', x='X Data',
                           y="Y")    # pick 250 sample randomly

plt.show()

batch_size = 8
# pick slope and intercept randomly as the start
m = tf.Variable(0.8)
b = tf.Variable(0.17)

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m * xph + b
error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_index = np.random.randint(len(x_data), size=batch_size)
        feed = {xph: x_data[rand_index], yph: y_true[rand_index]}
        sess.run(train, feed_dict=feed)
    model_m, model_b = sess.run([m, b])

# target m = 0.5, b= 5
print(model_m, model_b)

y_hat = x_data * model_m + model_b
my_data.sample(250).plot(kind='scatter', x= "X Data", y='Y')
plt.plot(x_data, y_hat, 'r' )
plt.show()