import tensorflow as tf
import numpy as np

a=tf.constant([[5.,8.],[4.,2.]])
b=tf.constant([3,3,3])

c= tf.reduce_mean(a)

sess =tf.Session()
#sess.run(c)

print(sess.run(c))