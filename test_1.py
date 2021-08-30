import tensorflow as tf #导入
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)  #astype 转化数组类型，转成float32
y_data = x_data*0.1 +0.3


###create tensorflow structure end###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))  #Variable 生成变量  【1】是shape 必须要， 后面的-1～1是范围，可以不要，
#这个范围与最后输出没有关系  后面那个shape是张量的形状
'''TensorFlow包含构建数据流图与计算数据流图等基本步骤，图中的节点表示数学操作，
图中连结各节点的边表示多维数组，即：tensors（张量）。 张量是TensorFlow最核心的组件，所有运算和优化都是基于张量进行的。
张量是基于向量和矩阵的推广，可以将标量看为零阶张量，矢量看做一阶张量，矩阵看做二阶张量。

上面是张量的标准解释，但是就我个人阅读完后看来，我觉得可以把张量看成数组，比如一阶张量就是一组数，
0阶就是一个数，虽然感觉有点部队，但是可以解释的通，同理，不同张量之间好像不能计算，目前还没有实验出来，定义也不是很详细
'''

biases =tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss =tf.reduce_mean(tf.square(y-y_data))

optimizer =tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
###create tensorflow structure end###

sess = tf.Session()
sess.run(init) #Very important
for step in range (201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))


