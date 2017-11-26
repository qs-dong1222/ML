import tensorflow as tf

# 指定列数就可以自动推导出行数，所以行数为None即可
input1 = tf.placeholder(tf.float32,[None,3])
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))

    print(sess.run(input1, feed_dict={input1:  ( [2,3,4],[7,8,9] )          }))

