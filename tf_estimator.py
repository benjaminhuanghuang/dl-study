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

# plt.show()

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
my_data.sample(250).plot(kind='scatter', x="X Data", y='Y')
# plt.plot(x_data, y_hat, 'r' )
# plt.show()


feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
#The Estimator object wraps a model which is specified by a model_fn, 
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split
# split data into training data, and testing data
x_train, x_eval, y_train, y_eval = train_test_split(
    x_data, y_true, test_size=0.3, random_state=101)
print(x_train.shape)  # 7000
print(x_eval.shape)  # 3000

input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=8, num_epochs=1000, shuffle=False)

eval_input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval}, y_eval, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

print('Training data metrics')
print(train_metrics)
print('Eval data metrics')
print(eval_metrics)



brand_new_data = np.linspace(0,10,10)
input_fun_predict = tf.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

predictions = []
for pred in estimator.predict(input_fn=input_fun_predict):
    predictions.append(pred['predictions'])


my_data.sample(n=250).plot(kind='scatter', x= 'X Data', y='Y' )
plt.plot(brand_new_data, predictions, 'r*')
plt.show()