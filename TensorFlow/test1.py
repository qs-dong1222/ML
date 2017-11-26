import tensorflow as tf
import numpy as np

# create feature data and real output
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3


##########   tensorflow structure   ##########
# 1 demension, from -1~1
# demension derives from the number of features
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
# 1 demension
biases = tf.Variable(tf.zeros([1]))
# initializing action
initAction = tf.global_variables_initializer()
# predict expression
y = Weights*x_data + biases
# cost function
cost = tf.reduce_mean(tf.square(y-y_data))
# learning rate 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
trainAction = optimizer.minimize(cost)
##########   tensorflow structure   ##########


sess = tf.Session()
# do initialization
sess.run(initAction)


print("init weight:",sess.run(Weights),"init bias:",sess.run(biases))
for epoch in range(201):
    # do train once
    sess.run(trainAction)
    if(epoch%20 ==0):
        print("epoch:",epoch, "weight:",sess.run(Weights),"bias:",sess.run(biases))
