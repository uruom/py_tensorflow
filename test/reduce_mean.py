import tensorflow as tf
import numpy as np

a=tf.constant([[5.,8.],[4.,2.]])
b=tf.constant([3,3,3])

c= tf.reduce_mean(a)

sess =tf.Session()
#sess.run(c)

print(sess.run(c))  #取这个里面所有数的平均值，而且直接输出c的话，是c这个张量，输出的并不是值，必须药用sess.run(c)才有用
'''输出的值有不同类型的区别，比如b中是3，3，4， 那么结果是3。因为是int出来的，但是如果是3.，3.，4.，那么结果是3.33333，因为后面的点表示是float
在b中，如果只有一个是float，那么最终结果也会有小数，不知道是因为里面有一个是float就默认了统一张量中全是float，还是是因为只要有一个，那么最后sess
出来的就是float
