import numpy as np



# class Batch:
#     def __init__(self, featureSet, labelSet):
#         if (featureSet.shape[0] != labelSet.shape[0]):
#             print("featureSet number != labelSet number");
#             print("featureSet size, ", featureSet.shape)
#             print("labelSet size, ", labelSet.shape)
#             return None;
#         else:
#             self.featureSet = featureSet;
#             self.labelSet = labelSet;
#             self.endIdx = self.featureSet.shape[0] - 1;
#             self.currIdx = 0;
#
#     def next_batch(self, batchNbr, sequentialBatch=True):
#         if (batchNbr > self.featureSet.shape[0]):
#             batchNbr = self.featureSet.shape[0]
#
#         if (sequentialBatch):
#             lastIdx = self.currIdx + batchNbr - 1;
#             if (lastIdx < self.endIdx):
#                 retFeatureBatch = self.featureSet[self.currIdx:lastIdx + 1];
#                 retLabelBatch = self.labelSet[self.currIdx:lastIdx + 1];
#                 self.currIdx = lastIdx + 1;
#                 return retFeatureBatch, retLabelBatch;
#             elif (lastIdx == self.endIdx):
#                 retFeatureBatch = self.featureSet[self.currIdx:lastIdx + 1];
#                 retLabelBatch = self.labelSet[self.currIdx:lastIdx + 1];
#                 self.currIdx = 0;
#                 return retFeatureBatch, retLabelBatch;
#             else:
#                 retFeatureBatch = self.featureSet[self.currIdx:self.endIdx + 1];
#                 retLabelBatch = self.labelSet[self.currIdx:self.endIdx + 1];
#                 self.currIdx = 0;
#                 return retFeatureBatch, retLabelBatch;
#         else:
#             idxes = np.random.randint(0, self.endIdx + 1, batchNbr);
#             retFeatureBatch = self.featureSet[idxes, :];
#             retLabelBatch = self.labelSet[idxes, :];
#             return retFeatureBatch, retLabelBatch;



class LinearRegressionOptimizer:
    def __init__(self, featureSet, labelSet):
        self.__LoadData__(featureSet, labelSet)


    def __LoadData__(self, featureSet, labelSet):
        self.xdata = np.array(featureSet)
        if(self.xdata.ndim<2):
            self.xdata = self.xdata.reshape(1, self.xdata.shape[0])
        self.ydata = np.array(labelSet)
        if (self.ydata.ndim < 2):
            self.ydata = self.ydata.reshape(1, self.ydata.shape[0])
        n_feature = self.xdata.shape[1] # number of cols in xdata
        initOffset = np.random.randint(low=0,high=10,size=n_feature) # randint 0~9
        self.w = np.random.randn(n_feature) + initOffset # randint 0~9 + fraction[0~1]
        self.w = self.w.reshape(1,n_feature)
        self.b = np.random.randint(0,10);


    def __GetRegularizationTerm__(self, regArg=0):
        return regArg * np.sum(np.square(self.w))


    def fit(self, learningRate=0.02, epoch=2000, batchSize=100, regLambda=0, showLoss=False):
        #batchSet_XY = Batch(self.xdata, self.ydata);
        for i in range(epoch):
            #batch_xs, batch_ys = batchSet_XY.next_batch(batchSize);
            #self.Train(batch_xs, batch_ys, learningRate, regLambda)
            self.Train(self.xdata, self.ydata, learningRate, regLambda)
            if (i%50==0 and showLoss):
                print(self.GetLoss(self.xdata, self.ydata))


    def predict(self, featureSet):
        hypoVals = np.dot(featureSet, self.w.T) + self.b
        return hypoVals


    def GetLoss(self, featureSet, labelSet):
        # loss => mean square error
        predVals = self.predict(featureSet)
        error = labelSet.T - predVals
        squreErr = np.square(error)
        sqErrSum = np.sum(squreErr, axis=0) # vertical sum
        currloss = 0.5 * (sqErrSum + self.__GetRegularizationTerm__(self.regArg)) # 0.5 * prediction loss + regularization
        return currloss;


    def Train(self, featureSet, labelSet, learningRate=0.02, regLambda=0):
        self.regArg = regLambda
        error = labelSet.T - self.predict(featureSet)

        deriv_w = np.mean(featureSet * (-1) * error, axis=0)
        self.w = (1-self.regArg)*self.w - learningRate*deriv_w
        deriv_b = np.mean((-1) * error, axis=0)
        self.b = self.b - learningRate*deriv_b


    def GetAccuracy(self, featureSet, labelSet):
        x = np.array(featureSet)
        if(x.ndim<2):
            x = x.reshape(1, x.shape[0])
        y = np.array(labelSet)
        if(y.ndim < 2):
            y = y.reshape(1, y.shape[0])
        error = y.T - self.predict(featureSet)
        acc = (1 - np.max(np.abs(error / y.T), axis=0)) * 100
        return acc










x = np.random.randint(10,size=200).reshape(100,2)
y = [i for i in range(1,21)]
y = y*5

x = np.arange(1,51).reshape(10,5)
y = [10,20,30,40,50,60,70,80,90,100]

lr = LinearRegressionOptimizer(x, y)
lr.fit(0.0001, 18000)

print("acc = ",lr.GetAccuracy(x,y))

print("ans = \n", lr.predict(x))








