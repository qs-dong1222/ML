from MyAPIs import *;
import cv2;










img1 = cv2.imread("1.jpg");
img2 = cv2.imread("2.jpg");
img3 = cv2.imread("3.jpg");
featureSet = np.array([ConvertImg2List(img1),ConvertImg2List(img2),ConvertImg2List(img3)]);
labelSet = np.array([ [0,0,1], [1,0,0], [0,1,0] ]);
x_val = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
imgSet = tf.reshape(x_val,[-1,50,50,3])
y_val = tf.placeholder(tf.float32, [None, labelSet.shape[1]])


conv_L1 = AddConv2dLayer(imgSet,[5,5],4,[1,1],'SAME')
maxpool_L1 = AddMaxPooling(conv_L1,[2,2],[2,2],'SAME');
conv_L2 = AddConv2dLayer(maxpool_L1,[5,5],8,[1,1],'SAME')
maxpool_L2 = AddMaxPooling(conv_L2,[2,2],[2,2],'SAME');

NN_in = FlattenMaxPool(maxpool_L2)

L1 = AddLayer(NN_in,10,tf.nn.relu);
predict_outputs = AddLayer(L1,3,None);


loss = tf.losses.softmax_cross_entropy(onehot_labels=y_val,logits=predict_outputs)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    batchSet_XY = Batch(featureSet, labelSet);
    for epoch in range(2000):
        batch_xs, batch_ys = batchSet_XY.next_batch(1);
        sess.run(train_step,feed_dict={x_val: batch_xs, y_val: batch_ys});
        if(epoch % 100 == 0):
            loss_training = GetCurrentLoss(sess,loss,x_val,featureSet,y_val,labelSet);
            print("loss_training", loss_training);

    result = GetClassifierAccuracy(sess
                                   , predict_outputs
                                   , x_val
                                   , featureSet
                                   , labelSet);
    print("acc =",result)









# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets(" MNIST_data/", one_hot=True)
# featureSet = mnist.train.images
# labelSet = mnist.train.labels
# x_val = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
# imgSet = tf.reshape(x_val,[-1,28,28,1])
# y_val = tf.placeholder(tf.float32, [None, labelSet.shape[1]])
#
#
#
#
# conv_L1 = AddConv2dLayer(imgSet,[5,5],4,[1,1],'SAME')
# maxpool_L1 = MaxPooling(conv_L1,[2,2],[2,2],'SAME');
# conv_L2 = AddConv2dLayer(maxpool_L1,[5,5],8,[1,1],'SAME')
# maxpool_L2 = MaxPooling(conv_L2,[2,2],[2,2],'SAME');
#
# NN_in = FlattenMaxPool(maxpool_L2)
#
# L1 = AddLayer(NN_in,10,tf.nn.relu);
# predict_outputs = AddLayer(L1,10,None);
#
#
# loss = tf.losses.softmax_cross_entropy(onehot_labels=y_val,logits=predict_outputs)
# train_step = tf.train.AdamOptimizer(0.005).minimize(loss);
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer());
#     batchSet_XY = Batch(featureSet, labelSet);
#     for epoch in range(2000):
#         batch_xs, batch_ys = batchSet_XY.next_batch(100);
#         # batch_xs, batch_ys = mnist.train.next_batch(100);
#         sess.run(train_step,feed_dict={x_val: batch_xs, y_val: batch_ys});
#         if(epoch % 100 == 0):
#             loss_training = GetCurrentLoss(sess,loss,x_val,mnist.test.images,y_val,mnist.test.labels);
#             print("loss_training", loss_training);
#
#     result = GetClassifierAccuracy(sess
#                                    , predict_outputs
#                                    , x_val
#                                    , mnist.test.images
#                                    , mnist.test.labels);
#     print("acc =",result)









