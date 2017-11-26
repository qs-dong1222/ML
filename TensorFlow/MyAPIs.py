import tensorflow as tf;
import numpy as np;
import cv2;

def AddLayer(inputDataSet, layerNeuroNbr, activationFunc=None, dropout_keep_prob=1, layerName='NNlayer'):
    with tf.name_scope(layerName):
        rowNbr = int(inputDataSet.shape[1]);
        Weights = tf.Variable(tf.random_normal([rowNbr,layerNeuroNbr]),name=layerName+"_W");
        b = tf.Variable(tf.zeros([1, layerNeuroNbr]) + 0.1, name=layerName+"_b");
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
                          , test_label_set):
    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(predictFunc, 1), tf.argmax(test_label_set, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return session.run(accuracy, feed_dict={feature_holder:test_feature_set})

def GetCurrentLoss(session, lossFunc
                   , feature_holder
                   , test_feature_set
                   , label_holder
                   , test_label_set):
    with tf.name_scope('loss'):
        curr_loss = session.run(lossFunc
                               ,feed_dict={label_holder: test_label_set, feature_holder: test_feature_set});
        return curr_loss;





def Conv2d(imageSet, kernelSize, outputChNbr, paceLength, paddingMode):
    # image set 有一堆独立的个体(图片), 每个图片可再细分层为一堆薄图片, n_ch就代表细分后的薄图片个数
    try:
        n_ch = int(imageSet[0].shape[2]);
    except:
        n_ch = int(1);
    # print("n_ch=",n_ch)

    kernel_w = kernelSize[0];
    kernel_h = kernelSize[1];
    # 制作一个n_ch层厚度的一个卷积核, 之后用此卷积核同时对n_ch层薄图片进行卷积操作, 将n_ch层变成outputChNbr层
    shape = [kernel_h,kernel_w,n_ch,int(outputChNbr)]
    kernel = tf.Variable(tf.truncated_normal(shape, stddev=0.1),name="conv_kernel");

    pace_horizontal = paceLength[0];
    pace_vertical = paceLength[1];
    stride = [1,pace_horizontal,pace_vertical,1];

    return tf.nn.conv2d(imageSet, kernel, strides=stride, padding=paddingMode);



def AddConv2dLayer(imageSet, kernelSize, outputChNbr, paceLength, paddingMode, activationFunc=None, layerName='conv2d'):
    with tf.name_scope(layerName):
        convRes = Conv2d(imageSet, kernelSize, outputChNbr, paceLength, paddingMode);
        b = tf.Variable(tf.zeros([1, int(convRes.shape[3])]) + 0.1,name=layerName+"_b");
        conv2d_plus_b = convRes + b;
        if activationFunc is None:
            layerOutputs = tf.nn.relu(conv2d_plus_b);
        else:
            layerOutputs = activationFunc(conv2d_plus_b)
        return layerOutputs;




def AddMaxPooling(convLayerOutput,kernelSize, paceLength, paddingMode, layerName='maxpooling'):
    with tf.name_scope(layerName):
        kernel_w = kernelSize[0];
        kernel_h = kernelSize[1];

        pace_horizontal = paceLength[0];
        pace_vertical = paceLength[1];
        stride = [1, pace_horizontal, pace_vertical, 1];

        return tf.nn.max_pool(convLayerOutput,ksize=[1,kernel_h,kernel_w,1],strides=stride,padding=paddingMode);



def FlattenMaxPool(maxpool_output):
    h, w, n_ch = maxpool_output.shape[1:];
    flat = tf.reshape(maxpool_output, [-1, int(h * w * n_ch)]);
    return flat



def ConvertImg2List(img):
    if (len(img.shape)>2):
        ch1, ch2, ch3 = cv2.split(img);
        sample = [];
        ch1 = ch1.flatten();
        for x in ch1:
            sample.append(x)
        ch2 = ch2.flatten();
        for x in ch2:
            sample.append(x)
        ch3 = ch3.flatten();
        for x in ch3:
            sample.append(x)
    else:
        sample = img.flatten();
    return sample








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


