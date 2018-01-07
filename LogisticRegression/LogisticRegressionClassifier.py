from Batch import *
import sys
import math



def Wx_plus_b(W, x, b):
    W = numpilize(W)
    x = numpilize(x)
    ans = np.dot(W, x.T) + b
    return ans



def Sigmoid(z):
    denom = 1 + np.exp(-z)
    ans = 1/denom
    return ans





kkk = 6


class LogisticRegssionClassifier:
    def __init__(self, featureSet, labelSet):
        self.xdata = numpilize(featureSet)
        self.ydata = numpilize(labelSet)
        self.n_sample = self.xdata.shape[0]
        if(self.n_sample != self.ydata.shape[1]):
            sys.stderr.write("Error!!!: feature size != label size\n")
            return None
        else:
            self.n_feature = self.xdata.shape[1]
            initOffset = np.random.randint(low=-1, high=2, size=self.n_feature)  # randint offset -5~5
            self.w = np.random.randn(self.n_feature) + initOffset  # fraction[0~1] + offset -5~5
            self.w = numpilize(self.w)
            self.b = np.random.randint(-1, 2);
            self.regArg = 0


    def predict(self, featureSet):
        xs = numpilize(featureSet)
        z = Wx_plus_b(self.w, xs, self.b)
        ans = Sigmoid(z)
        return ans


    def GetLoss(self, featureSet, labelSet):
        xs = numpilize(featureSet)
        ys = numpilize(labelSet)
        hypos = self.predict(xs)
        log_hypos = np.log(hypos)
        log_hypos = np.where(log_hypos==np.nan, 0, log_hypos)
        log_1_minus_hypos = np.log(1-hypos)
        log_1_minus_hypos = np.where(log_1_minus_hypos==np.nan, 0, log_1_minus_hypos)
        entropy_loss = -1 * np.sum(ys * log_hypos + (1-ys) * log_1_minus_hypos)
        regularization = 0.5 * self.regArg * np.sum(np.square(self.w))
        totLoss = entropy_loss + regularization
        return totLoss


    def Train(self, featureSet, labelSet, learningRate=0.01, regLambda=0, enable_ada=False):
        ######################
        ### Adagrad method ###
        ######################
        xs = numpilize(featureSet)
        ys = numpilize(labelSet)
        hypos = self.predict(xs)
        error = ys - hypos
        self.regArg = regLambda
        self.ada_iter += 1
        # derivative of w
        deriv_w = np.mean(xs * (-1) * error.T, axis=0)
        self.deriv_w_sqsum += np.square(deriv_w)
        ada_w = np.sqrt(self.deriv_w_sqsum / self.ada_iter)
        ada_w = np.where(ada_w == 0, 1e-7, ada_w)
        # derivative of b
        deriv_b = (-1) * np.mean(error, axis=1)
        self.deriv_b_sqsum += np.square(deriv_b)
        ada_b = np.sqrt(self.deriv_b_sqsum / self.ada_iter)
        ada_b = np.where(ada_b == 0, 1e-7, ada_b)
        # update w and b
        if(enable_ada):
            self.w = (1 - self.regArg) * self.w - (learningRate / ada_w) * deriv_w
            self.b = self.b - (learningRate/ada_b)*deriv_b
        else:
            self.w = (1 - self.regArg) * self.w - learningRate * deriv_w
            self.b = self.b - learningRate * deriv_b



    def fit(self, learningRate=0.01, epoch=2000, batchSize=100, regLambda=0, enable_ada=False, showLoss=False):
        self.deriv_w_sqsum = np.zeros([1, self.xdata.shape[1]])
        self.deriv_b_sqsum = 0
        self.ada_iter = 0

        batchSet_XY = Batchor(self.xdata, self.ydata);
        for i in range(epoch):
            batch_xs, batch_ys = batchSet_XY.NextBatch(batchSize);
            self.Train(batch_xs, batch_ys, learningRate, regLambda, enable_ada=enable_ada)
            if (i%50==0 and showLoss):
                print(self.GetLoss(self.xdata, self.ydata))







# x = np.arange(1000,1100).reshape(50,2)/20
# y = np.random.randint(0,2,size=50)
#
#
# lrc = LogisticRegssionClassifier(x, y)
# lrc.fit(learningRate= 0.01, epoch=5000, batchSize=50,showLoss=True)


