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





class LogisticRegssionClassifier:
    """
        feature set should follow input format below:
                   x1 x2 x3 ... xn
        line No.1  a  b  c  ... k
        line No.2  m  q  h  ... w
           ...     .  .  .  ... .
           ...     .  .  .  ... .
        line No.k  r  s  t  ... j


        label set should follow input format below:
                   y1 y2 y3 y4 ... yn -> single line data
        line No.1  a  b  c  d  ... k
    """
    def __init__(self, featureSet, labelSet):
        self.xdata = numpilize(featureSet)
        self.ydata = numpilize(labelSet)
        self.n_sample = self.xdata.shape[0]
        if(self.n_sample != self.ydata.shape[1]):
            sys.stderr.write("Error!!!: feature size != label size\n")
            return None
        else:
            self.n_feature = self.xdata.shape[1]
            init_w_Offset = np.random.rand(self.n_feature)  # randint offset -5~5
            self.w = np.random.randn(self.n_feature) + init_w_Offset  # fraction[0~1] + offset -5~5
            self.w = numpilize(self.w)
            self.b = 1;
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
        hypos = np.where(hypos == 0, 1e-11, hypos)
        hypos = np.where(hypos == 1, 0.99999999, hypos)
        log_hypos = np.log(hypos)
        log_hypos = np.where(log_hypos==np.nan, 0, log_hypos)
        log_1_minus_hypos = np.log(1-hypos)
        log_1_minus_hypos = np.where(log_1_minus_hypos==np.nan, 0, log_1_minus_hypos)
        entropy_loss = -1 * np.sum(ys * log_hypos + (1-ys) * log_1_minus_hypos)
        regularization = 0.5 * self.regArg * np.sum(np.square(self.w))
        totLoss = (entropy_loss + regularization)/xs.shape[0]
        return totLoss



    def fit(self, epoch=2000, batchSize=100, regLambda=0,
            learningRate=0.001,
            mtum_beta=0.9,
            rmsprop_beta=0.999,
            epsilon=1e-8,
            showLoss=False):
        # ######################
        # ### Adagrad method ###
        # ######################
        # self.deriv_w_sqsum = np.zeros([1, self.xdata.shape[1]])
        # self.deriv_b_sqsum = 0

        ######################
        ###   Adam method  ###
        ######################
        self.mtum_w = np.zeros([1, self.xdata.shape[1]])
        self.rmsprop_w = np.zeros([1, self.xdata.shape[1]])
        self.mtum_b = 0
        self.rmsprop_b = 0
        self.Adam_iter = 0

        batchSet_XY = Batchor(self.xdata, self.ydata);
        for i in range(epoch):
            batch_xs, batch_ys = batchSet_XY.NextBatch(batchSize);
            self.Train(batch_xs, batch_ys, regLambda=regLambda,
                       learningRate=learningRate,
                       mtum_beta=mtum_beta,
                       rmsprop_beta=rmsprop_beta,
                       epsilon=epsilon)
            if (i%100==0 and showLoss):
                print(self.GetLoss(self.xdata, self.ydata))



    def Train(self, featureSet, labelSet, regLambda=0,
              learningRate=0.001,
              mtum_beta=0.9,
              rmsprop_beta=0.999,
              epsilon=1e-8
              ):
        # ######################
        # ### Adagrad method ###
        # ######################
        # xs = numpilize(featureSet)
        # ys = numpilize(labelSet)
        # hypos = self.predict(xs)
        # error = ys - hypos
        # self.regArg = regLambda
        # # derivative of w
        # deriv_w = np.mean(xs * (-1) * error.T, axis=0)
        # self.deriv_w_sqsum += np.square(deriv_w)
        # ada_w = np.sqrt(self.deriv_w_sqsum)
        # ada_w = np.where(ada_w == 0, 1e-7, ada_w)
        # # derivative of b
        # deriv_b = (-1) * np.mean(error, axis=1)
        # self.deriv_b_sqsum += np.square(deriv_b)
        # ada_b = np.sqrt(self.deriv_b_sqsum)
        # ada_b = np.where(ada_b == 0, 1e-7, ada_b)
        # # update w and b
        # if(enable_ada):
        #     self.w = (1 - self.regArg) * self.w - (learningRate / ada_w) * deriv_w
        #     self.b = self.b - (learningRate/ada_b)*deriv_b
        # else:
        #     self.w = (1 - self.regArg) * self.w - learningRate * deriv_w
        #     self.b = self.b - learningRate * deriv_b

        ######################
        ###   Adam method  ###
        ######################
        xs = numpilize(featureSet)
        ys = numpilize(labelSet)
        hypos = self.predict(xs)
        error = ys - hypos

        g_w = np.mean(xs * (-1) * error.T, axis=0)
        g_b = (-1) * np.mean(error, axis=1)
        self.regArg = regLambda
        self.Adam_iter += 1

        self.mtum_w = mtum_beta * self.mtum_w + (1 - mtum_beta) * g_w
        self.rmsprop_w = rmsprop_beta * self.rmsprop_w + (1 - rmsprop_beta) * np.square(g_w)
        bias_mtum_w = 1 - np.power(mtum_beta, self.Adam_iter)
        bias_rmsprop_w = 1 - np.power(rmsprop_beta, self.Adam_iter)
        bias_correct_mtum_w = self.mtum_w / bias_mtum_w
        bias_correct_rmsprop_w = self.rmsprop_w / bias_rmsprop_w

        self.mtum_b = mtum_beta * self.mtum_b + (1 - mtum_beta) * g_b
        self.rmsprop_b = rmsprop_beta * self.rmsprop_b + (1 - rmsprop_beta) * np.square(g_b)
        bias_mtum_b = 1 - np.power(mtum_beta, self.Adam_iter)
        bias_rmsprop_b = 1 - np.power(rmsprop_beta, self.Adam_iter)
        bias_correct_mtum_b = self.mtum_b / bias_mtum_b
        bias_correct_rmsprop_b = self.rmsprop_b / bias_rmsprop_b

        self.w = (1 - learningRate*self.regArg) * self.w - learningRate * bias_correct_mtum_w / (np.sqrt(bias_correct_rmsprop_w) + epsilon)
        self.b = self.b - learningRate * bias_correct_mtum_b / (np.sqrt(bias_correct_rmsprop_b) + epsilon)












# x = np.arange(1000,1100).reshape(50,2)/20
# y = np.random.randint(0,2,size=50)
#
#
# lrc = LogisticRegssionClassifier(x, y)
# lrc.fit(epoch=5000, batchSize=200,regLambda=0,
#         learningRate=0.001,
#         mtum_beta=0.9,
#         rmsprop_beta=0.999,
#         showLoss=True)




















