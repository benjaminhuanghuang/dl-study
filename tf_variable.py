import tensorflow as tf

const = tf.constant(10)
fill_mat = tf.fill((4, 4), 10)
myzeros = tf.zeros((4,4))
myones = tf.ones((4,4))
myrandoms = tf.random_normal((10,4), mean=0, stddev=1.0)
myrandoms2 = tf.random_uniform((10,4), minval=0, maxval=1.0)
myops= [const, fill_mat, myzeros, myones]

with tf.Session() as sess:
    for op in myops:
        print(sess.run(op))  # op.eval()


a = tf.constant([
    [1,2,6],
    [3,4,7]
])

print(a.get_shape())

b = tf.constant([
    [110],
    [111],
    [121]
])

op = tf.matmul(a,b)
with tf.Session() as sess:
    result = sess.run(op)
print(result)