from Batch import *
from math import *

class NaiveBayesClassifier:
    """
        feature set should follow input format below:
                   x1 x2 x3 ... xn
        line No.1  a  b  c  ... k
        line No.2  m  q  h  ... w
           ...     .  .  .  ... .
           ...     .  .  .  ... .
        line No.k  r  s  t  ... j
    """
    def __init__(self, *featureSets):
        self.n_classes = len(featureSets)
        self.ClassProbCalcFunc = []
        # build probability model for each class according to feature sets of different classes
        for i in range(self.n_classes):
            self.ClassProbCalcFunc.append( self.BuildNBCmodel( featureSets[i] ) )


    # predict class label given by a single sample data
    def PredictClass(self, sample):
        probs = self.ClassProbabilityCalc(sample)
        return np.argmax(probs)



    # calculate probablity to each class given by a single sample data
    def ClassProbabilityCalc(self, sample):
        probablity_list = [calcFunc(sample) for calcFunc in self.ClassProbCalcFunc]
        np.clip(probablity_list, 1e-11, 0.9999999999)
        return probablity_list





    def BuildNBCmodel(self, featureSet):
        xs = numpilize(featureSet)
        mu = np.mean(xs, axis=0)
        std_devia = np.mean(np.square(xs - mu), axis=0)
        # define a nested function that returns the probability to
        # the current class belongs to the given feature set, given a input sample.
        # principle is based on naive bayes model,
        # assuming each feature is independent from another
        def NbcProbabilityCalc(sample):
            ans = 1
            for i in range(xs.shape[1]):
                ans *= self.Gaussian(mu[i], std_devia[i], sample[i])
            return ans
        # return the defined function
        return NbcProbabilityCalc




    # guassian function to compute a probablity of x, based on the given parameters
    # of mu and std_devia
    def Gaussian(self, mu, std_devia, x):
        return (1 / (sqrt(2 * pi) * std_devia) * e ** (-0.5 * (float(x - mu) / std_devia) ** 2))











x1 = np.random.randint(1,100,300).reshape(100,3)
x2 = np.random.randint(50,150,360).reshape(120,3)
x3 = np.random.randint(100,200,240).reshape(80,3)

nbc = NaiveBayesClassifier(x1, x2, x3)


print(nbc.ClassProbabilityCalc(x1[2]))

