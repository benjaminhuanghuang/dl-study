import tensorflow as tf

print (tf.__version__)

hello = tf.constant('Hello ')
world = tf.constant('World')
print(type(hello))
print(hello)    # tenser

with tf.Session() as sess:
    result = sess.run(hello + world)

print(result)



a = tf.constant(10)
b = tf.constant(11)
with tf.Session() as sess:
    result = sess.run(a + b)

print(result)