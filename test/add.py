import tensorflow as tf

a=tf.constant([1]) +2
b=tf.constant([2])

c=tf.add(a,b)
with tf.Session() as sess:
    
    print(sess.run(c))