import tensorflow as tf

matrix1 =tf.constant([[3,3]])
matrix2 =tf.constant([[2],[2]])#最后答案是[[12]],其实就是矩阵乘法，(3,3)* 2  最后答案就是12
#                                                                         2  
product = tf.matmul(matrix1,matrix2)  #matrix multiply np.dot(m1,m2)

###method 1 两个不同的输出方式
#sess= tf.Session()
#result =sess.run(product)
#print(result)
#sess.close()

#method 2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)


