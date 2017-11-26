import numpy as np;
import tensorflow as tf;

mat1 = tf.constant([ [3,3] ]);
mat2 = tf.constant([ [2],[2] ]);
matmulAction = tf.matmul(mat1,mat2);

#### method 1
# sess = tf.Session();
# product = sess.run(matmulAction);
# print(product)
# sess.close();


#### method 2
with tf.Session() as sess:
    product = sess.run(matmulAction)
    print(product)

