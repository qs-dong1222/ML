# when 70 nuerons in 1st layer, 50 nuerons in 2nd layer, 10 nuerons in output layer,
# appling dropout of 0.2(keep of 0.8), the accuracy can reach to 95.8% after 40 epoch

import tensorflow as tf
import numpy as np;
import os;
import glob;
import matplotlib.pyplot as plt



def AddLayer(inputDataSet, layerNeuroNbr, activationFunc=None, dropout_keep_prob=1):
    rowNbr = int(inputDataSet.shape[1]);
    Weights = tf.Variable(tf.random_normal([rowNbr,layerNeuroNbr]));
    b = tf.Variable(tf.zeros([1, layerNeuroNbr]) + 0.1);
    Wx_plus_b = tf.matmul(inputDataSet, Weights) + b;
    if activationFunc is None:
        layerOutputs = Wx_plus_b
    else:
        layerOutputs = activationFunc(Wx_plus_b)
    layerOutputs = tf.nn.dropout(layerOutputs, dropout_keep_prob);
    return layerOutputs;

def GetClassifierAccuracy(session, predictFunc
                          , feature_holder
                          , test_feature_set
                          , test_label_set
                          , drop_keep_prob_holder
                          , drop_keep_val):
    correct_prediction = tf.equal(tf.argmax(predictFunc, 1), tf.argmax(test_label_set, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return session.run(accuracy, feed_dict={feature_holder:test_feature_set
                                          , drop_keep_prob_holder:drop_keep_val});

def GetCurrentLoss(session, lossFunc
                   , feature_holder
                   , test_feature_set
                   , label_holder
                   , test_label_set
                   , drop_keep_prob_holder
                   , drop_keep_val):
    curr_loss = session.run(lossFunc
                           ,feed_dict={label_holder: test_label_set
                                     , feature_holder: test_feature_set
                                     , drop_keep_prob_holder: drop_keep_val});
    return curr_loss;


class Batch:
    def __init__(self, featureSet, labelSet):
        if (featureSet.shape[0] != labelSet.shape[0]):
            print("featureSet number != labelSet number");
            return None;
        else:
            self.featureSet = featureSet;
            self.labelSet = labelSet;
            self.endIdx = self.featureSet.shape[0] - 1;
            self.currIdx = 0;

    def next_batch(self, batchNbr, sequentialBatch=True):
        if (batchNbr > self.featureSet.shape[0]):
            print("batchNbr is too large");
            return None;

        if (sequentialBatch):
            lastIdx = self.currIdx + batchNbr - 1;
            if (lastIdx < self.endIdx):
                retFeatureBatch = self.featureSet[self.currIdx:lastIdx + 1];
                retLabelBatch = self.labelSet[self.currIdx:lastIdx + 1];
                self.currIdx = lastIdx + 1;
                return retFeatureBatch, retLabelBatch;
            elif (lastIdx == self.endIdx):
                retFeatureBatch = self.featureSet[self.currIdx:lastIdx + 1];
                retLabelBatch = self.labelSet[self.currIdx:lastIdx + 1];
                self.currIdx = 0;
                return retFeatureBatch, retLabelBatch;
            else:
                retFeatureBatch = self.featureSet[self.currIdx:self.endIdx + 1];
                retLabelBatch = self.labelSet[self.currIdx:self.endIdx + 1];
                self.currIdx = 0;
                return retFeatureBatch, retLabelBatch;
        else:
            idxes = np.random.randint(0, self.endIdx + 1, batchNbr);
            retFeatureBatch = self.featureSet[idxes, :];
            retLabelBatch = self.labelSet[idxes, :];
            return retFeatureBatch, retLabelBatch;





from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(" MNIST_data/", one_hot=True)
featureSet = mnist.train.images
labelSet = mnist.train.labels





features_var = tf.placeholder(tf.float32, [None, featureSet.shape[1]])
labels_var = tf.placeholder(tf.float32, [None, labelSet.shape[1]])
drop_keep_prob_var = tf.placeholder(tf.float32)


L1_outputs = AddLayer(features_var, 70, tf.nn.relu, drop_keep_prob_var);
L2_outputs = AddLayer(L1_outputs, 50, tf.nn.relu, drop_keep_prob_var);
predict_outputs = AddLayer(L2_outputs, 10, None);


# 要想使用 tf.losses.softmax_cross_entropy(), 需要把输出层的activation function改为None, 因为API会自己加softmax
loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_var,logits=predict_outputs)
# loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_var, logits=predict_outputs)
# 要想使用 自己定义的cross_entropy函数, 需要把输出层的activation function改为tf.nn.softmax, 人为加上去
# def clip(x):
#     return tf.clip_by_value( x ,1e-10,1.0);
#### cross_entropy ####
# loss = -tf.reduce_mean(labels_var * tf.log(clip(predict_outputs)),axis=0)
#### logistic cross_entropy ####
# loss = -tf.reduce_mean(labels_var * tf.log(clip(predict_outputs)) + (1-labels_var) * tf.log(clip(1-predict_outputs)),axis=0)
#### square mean error ####
# loss = tf.reduce_mean(tf.square(labels_var-predict_outputs),axis=0);

train_step = tf.train.AdamOptimizer(0.004).minimize(loss);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());

    # fw = tf.summary.FileWriter('D:/mmm');
    # fw.add_graph(sess.graph)

    batchSet_XY = Batch(featureSet, labelSet);
    batch_size = 100;
    for epoch in range(1):
        for eachBatch in range( round(len(featureSet)/batch_size) ):
            batch_xs, batch_ys = batchSet_XY.next_batch(batch_size);
            # batch_xs, batch_ys = mnist.train.next_batch(100);
            sess.run(train_step,feed_dict={features_var: batch_xs
                                           , labels_var: batch_ys
                                           , drop_keep_prob_var: 0.8});

        loss_training = GetCurrentLoss(sess
                                       ,loss
                                       ,features_var
                                       ,mnist.test.images
                                       ,labels_var
                                       ,mnist.test.labels
                                       ,drop_keep_prob_holder=drop_keep_prob_var
                                       ,drop_keep_val=1);
        print("loss_training", loss_training);

        result = GetClassifierAccuracy(sess
                                       , predict_outputs
                                       , features_var
                                       , mnist.test.images
                                       , mnist.test.labels
                                       , drop_keep_prob_holder=drop_keep_prob_var
                                       , drop_keep_val=1);
        print("acc =",result)


