import numpy as np;
import tensorflow as tf;

state = tf.Variable(0,name="counter");
print(state.name);
init = tf.global_variables_initializer();


const_one = tf.constant(1);

newVal = tf.add(state,const_one);
update = tf.assign(state,newVal)


with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update);
        print(sess.run(state))

