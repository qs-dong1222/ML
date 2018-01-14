import numpy as np
from Batch import *

def numpilize(array_x):
    x = np.array(array_x)
    if (x.ndim < 2):
        x = x.reshape(1, x.shape[0])
    return x



class LinearRegressionOptimizer:
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


    # mtum_beta goes bigger, parameters update bigger
    # rmsprop_beta goes bigger, parameters update smaller
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
            if (i%500==0 and showLoss):
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
        currloss /= featureSet.shape[0]
        return currloss;


    def Train(self, featureSet, labelSet, regLambda=0,
              learningRate=0.001,
              mtum_beta=0.9,
              rmsprop_beta=0.999,
              epsilon=1e-8,
              ):
        # ######################
        # ### Adagrad method ###
        # ######################
        # self.regArg = regLambda
        # error = labelSet.T - self.predict(featureSet)
        #
        # deriv_w = np.mean(featureSet * (-1) * error, axis=0)
        # self.deriv_w_sqsum  += np.square(deriv_w)
        # ada_w = np.sqrt(self.deriv_w_sqsum)
        # ada_w = np.where(ada_w == 0, 1e-7, ada_w)
        # self.w = (1-self.regArg)*self.w - (learningRate/ada_w)*deriv_w
        #
        # deriv_b = np.mean((-1) * error, axis=0)
        # self.deriv_b_sqsum += np.square(deriv_b)
        # ada_b = np.sqrt(self.deriv_b_sqsum)
        # ada_b = np.where(ada_b == 0, 1e-7, ada_b)
        # self.b = self.b - (learningRate/ada_b)*deriv_b

        ######################
        ###   Adam method  ###
        ######################
        self.regArg = regLambda
        error = labelSet.T - self.predict(featureSet)
        g_w = np.mean(featureSet * (-1) * error, axis=0)
        g_b = np.mean((-1) * error, axis=0)
        self.Adam_iter += 1

        self.mtum_w = mtum_beta*self.mtum_w + (1-mtum_beta)*g_w
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

        self.w = (1 - learningRate*self.regArg) * self.w - learningRate*bias_correct_mtum_w/(np.sqrt(bias_correct_rmsprop_w)+epsilon)
        self.b = self.b - learningRate*bias_correct_mtum_b/(np.sqrt(bias_correct_rmsprop_b)+epsilon)






    def GetAccuracy(self, featureSet, labelSet):
        x = numpilize(featureSet)
        y = numpilize(labelSet)
        y = np.where(y==0, 1e-7, y)
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
# lr.fit(epoch=10000, batchSize=100, regLambda=0,
#        learningRate=0.01,
#        mtum_beta=0.9,
#        rmsprop_beta=0.999,
#        showLoss=True)
#
# print("acc = ",lr.GetAccuracy(x,y))
# print("ans = \n", lr.predict(x))







