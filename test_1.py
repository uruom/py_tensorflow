import tensorflow as tf #导入
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)  #astype 转化数组类型，转成float32
y_data = x_data*0.1 +0.3


###create tensorflow structure end###
Weights = tf.Variable(tf.random_uniform([1],-10,10))  #Variable 生成变量   
biases =tf.Variable(tf.zeros([1])) #zeros 生成一个空的（所有值都是0的）张量，张量的大小形状在shape【】中

y = Weights*x_data + biases

loss =tf.reduce_mean(tf.square(y-y_data)) #取平均值  reduce_mean是必须的，但是square可以换成abs，不过会慢一点，最主要的目的应该还是取正值

optimizer =tf.train.GradientDescentOptimizer(0.5)#这个地方数字是数字越小，最后答案离近似值越近，但是会非常慢，也就是训练次数会翻好几倍，
#0.1和1的差距大概是在10倍左右
train = optimizer.minimize(loss)
#可以二合一为这个：train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.initialize_all_variables()#初始化值
###create tensorflow structure end###

sess = tf.Session()
sess.run(init) #Very important
for step in range (201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biases))
#tensorflow的语法感觉上是倒着来的，就是需要什么用什么，session是开始，结束到所不需要知道其他的时候，
#比如sess.run(train)那么回过头去看train=，然后发现loss，就看loss=，一步步退回去的感觉


