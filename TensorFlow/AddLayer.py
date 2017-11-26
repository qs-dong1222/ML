# 此段程序是用于进行线性回归实验
# nueral network 的本质实际上是各层的各个输入经加权求和后，再进行非线性化(activation function)输出.
# 在本例子中，只有一层hidden layer，一个输入点。那么每个神经元的输出就是 activationFunc(w_i*x)。
# 那么输出层的输入就是 对 activationFunc(w_1*x)，activationFunc(w_2*x)，activationFunc(w_3*x) ...... activationFunc(w_n*x)
# 的加权求和，即各个输出节点的操作就是
# out = outputActFunc( actFunc(w_1*x) + actFunc(w_2*x) + actFunc(w_3*x) +...+ actFunc(w_n*x) )
# 这里我们输出处不进行非线性化处理，即outputActFunc不使用。
# 最终，out = actFunc(w_1*x) + actFunc(w_2*x) + actFunc(w_3*x) +...+ actFunc(w_n*x)
# 所以最后我们的模拟回归函数实际上就是一堆actFunc(w_i*x)的求和，类似泰勒级数的思想

# 理论上来说，增加各层神经元个数 和 增加hidden层数都可以改善最终结果
# rele激活函数 可以使得曲线的拟合过程更有棱角
# softmax激活函数 可以使得曲线更圆滑

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def AddLayer(inputDataSet, layerNeuroNbr, activationFunc=None):
    rowNbr = int(inputDataSet.shape[1]);
    Weights = tf.Variable(tf.random_normal([rowNbr,layerNeuroNbr]));
    b = tf.Variable(tf.zeros([1, layerNeuroNbr]) + 0.1);
    Wx_plus_b = tf.matmul(inputDataSet, Weights) + b;
    if activationFunc is None:
        layerOutputs = Wx_plus_b
    else:
        layerOutputs = activationFunc(Wx_plus_b)
    return layerOutputs;


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






featureSet = np.linspace(-1,1,300)[:,np.newaxis];
noise = np.random.normal(0, 0.05, featureSet.shape);
labelSet = 1.3*np.power(featureSet,1)\
           +np.power(featureSet,2)\
           +2*np.power(featureSet,3)\
           -19*np.power(featureSet,6)\
           -0.5 \
           + noise;


Xholder = tf.placeholder(tf.float32, [None, featureSet.shape[1]]);
Yholder = tf.placeholder(tf.float32, [None, labelSet.shape[1]]);


# 定义网络结构
L1_outputs = AddLayer(Xholder, 10, tf.nn.relu);
L2_outputs = AddLayer(L1_outputs, 6, tf.nn.softmax);
predict_outputs = AddLayer(L2_outputs, 1, None);

# 定义loss function
loss = tf.reduce_mean(tf.square(Yholder - predict_outputs),axis=0) # axis=1 行操作， axis=0 列操作
# 这个loss function 不是交叉熵的形式。因为最终输出层不是一个分类形式的激活函数，
# 而是一个单纯的线性回归输出值，不存在梯度衰减问题，或者说不是分类问题，交叉熵并不适用
# 总结来说，交叉熵一般有两种形式，分别对应于sigmoid和softmax两个激活函数。
# softmax的交叉熵代价函数有另一个叫法：对数释然代价函数
# 具体情况看输出层的激活函数是什么, 在tf中的函数分别是:
# ！！！！！注意！！！！！: 这里的tensorflow自带cross_entropy函数，根据API解释，是会在log里面自动去求
# softmax()和sigmoid()的，所以最后一层要是想用这两激活函数时，网络中输出层不要加这两个函数，API会帮你加上

# for softmax() as output layer
# loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_var,logits=predict_outputs)
# for sigmoid/tanh as output layer
# loss = tf.losses.sigmoid_cross_entropy(onehot_labels=labels_var,logits=predict_outputs)

# deprecate usage(不建议使用)
# loss = -tf.reduce_sum(Yholder* tf.log(tf.clip_by_value( predict_outputs ,1e-10,1.0)) );



# 定义train step
trainStep = tf.train.AdamOptimizer(0.01).minimize(loss);




# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.xlim(np.min(featureSet), np.max(featureSet));
plt.ylim(np.min(labelSet), np.max(labelSet));
ax.scatter(featureSet, labelSet)
plt.ion()
plt.show()





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    batchSet_XY = Batch(featureSet, labelSet);
    for i in range(2000):
        batch_xs, batch_ys = batchSet_XY.next_batch(100);
        sess.run(trainStep, feed_dict={Xholder: batch_xs, Yholder: batch_ys})
        if i % 50 == 0:
            print("loss\n", sess.run(loss, feed_dict={Xholder: featureSet, Yholder: labelSet}))
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            prediction_value = sess.run(predict_outputs, feed_dict={Xholder: featureSet})
            # plot the prediction
            lines = ax.plot(featureSet, prediction_value, 'r-', lw=5)
            plt.pause(0.1)

    x = 0.2
    print("prediction("+str(x)+") =",sess.run(predict_outputs, feed_dict={Xholder: [[x]]}));
    real = (1.3*x) + pow(x,2) + (2*pow(x, 3)) - (19*pow(x, 6)) - 0.5;
    print("real("+str(x)+") =",real)
