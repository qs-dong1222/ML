import sys
import numpy as np

def numpilize(array_x):
    x = np.array(array_x)
    if (x.ndim < 2):
        x = x.reshape(1, x.shape[0])
    return x



class Batchor:
    """ feature set should follow input format below:
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
            sys.stderr.write("!!! Error !!! : number of feature samples not equals to number of label samples\n")
            return None
        else:
            self.curIdx = 0


    def NextBatch(self, batchSize):
        if(batchSize > self.n_sample):
            batchSize = self.n_sample
        endIdx = self.curIdx + batchSize
        if(endIdx > self.n_sample):
            n_part = self.n_sample - self.curIdx
            batchX_a = self.xdata[self.curIdx : self.curIdx + n_part]
            batchY_a = self.ydata[:, self.curIdx : self.curIdx + n_part]
            n_rest = batchSize - n_part
            self.curIdx = 0
            batchX_b, batchY_b = self.NextBatch(n_rest)
            batchX = np.vstack((batchX_a, batchX_b))
            batchY = np.hstack((batchY_a, batchY_b))
            return batchX, batchY
        else:
            batchX = self.xdata[self.curIdx: endIdx]
            batchY = self.ydata[:, self.curIdx: endIdx]
            if(endIdx == self.n_sample):
                self.curIdx = 0
            else:
                self.curIdx = endIdx
            return batchX, batchY








# x = np.arange(15).reshape(5,3)
# y = range(5)
#
# bat = Batchor(x,y)
