import numpy as np
from Batch import *

def numpilize(array_x):
    x = np.array(array_x)
    if (x.ndim < 2):
        x = x.reshape(1, x.shape[0])
    return x



class LinearRegressionOptimizer:
    def __init__(self, featureSet, labelSet):
        self.__LoadData__(featureSet, labelSet)


    def __LoadData__(self, featureSet, labelSet):
        self.xdata = numpilize(featureSet)
        self.ydata = numpilize(labelSet)
        n_feature = self.xdata.shape[1] # number of cols in xdata
        initOffset = np.random.randint(low=0,high=10,size=n_feature) # randint 0~9
        self.w = np.random.randn(n_feature) + initOffset # randint 0~9 + fraction[0~1]
        self.w = numpilize(self.w)
        self.b = np.random.randint(0,10);


    def __GetRegularizationTerm__(self, regArg=0):
        return regArg * np.sum(np.square(self.w))


    def fit(self, learningRate=0.02, epoch=2000, batchSize=100, regLambda=0, showLoss=False):
        self.deriv_w_sqsum = np.zeros([1, self.xdata.shape[1]])
        self.deriv_b_sqsum = 0
        self.ada_iter = 0

        batchSet_XY = Batchor(self.xdata, self.ydata);
        for i in range(epoch):
            batch_xs, batch_ys = batchSet_XY.NextBatch(batchSize);
            self.Train(batch_xs, batch_ys, learningRate, regLambda)
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
        ######################
        ### Adagrad method ###
        ######################
        self.regArg = regLambda
        error = labelSet.T - self.predict(featureSet)
        self.ada_iter += 1

        deriv_w = np.mean(featureSet * (-1) * error, axis=0)
        self.deriv_w_sqsum  += np.square(deriv_w)
        ada_w = np.sqrt(self.deriv_w_sqsum / self.ada_iter)
        ada_w = np.where(ada_w == 0, 0e-7, ada_w)
        self.w = (1-self.regArg)*self.w - (learningRate/ada_w)*deriv_w

        deriv_b = np.mean((-1) * error, axis=0)
        self.deriv_b_sqsum += np.square(deriv_b)
        ada_b = np.sqrt(self.deriv_b_sqsum / self.ada_iter)
        ada_b = np.where(ada_b == 0, 0e-7, ada_b)
        self.b = self.b - (learningRate/ada_b)*deriv_b


    def GetAccuracy(self, featureSet, labelSet):
        x = numpilize(featureSet)
        y = numpilize(labelSet)
        y = np.where(y==0, 0e-7, y)
        error = y.T - self.predict(featureSet)
        acc = (1 - np.max(np.abs(error / y.T), axis=0)) * 100
        return acc










# x = np.random.randint(10,size=200).reshape(100,2)
# y = [i for i in range(1,21)]
# y = y*5




# x = np.arange(1,51).reshape(10,5)
# y = [10,20,30,40,50,60,70,80,90,100]
#
# lr = LinearRegressionOptimizer(x, y)
# lr.fit(0.3, 2000, showLoss=True)
#
# print("acc = ",lr.GetAccuracy(x,y))
# print("ans = \n", lr.predict(x))







