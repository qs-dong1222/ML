from Batch import *

class PGMclassifier:
    """
    feature set should follow input format below:
                   x1 x2 x3 ... xn
        line No.1  a  b  c  ... k
        line No.2  m  q  h  ... w
           ...     .  .  .  ... .
           ...     .  .  .  ... .
        line No.k  r  s  t  ... j
    """
    def __init__(self, xs_class_1, xs_class_2):
        xs_class_1 = numpilize(xs_class_1)
        xs_class_2 = numpilize(xs_class_2)
        self.n_sample_c1 = xs_class_1.shape[0]
        self.n_sample_c2 = xs_class_2.shape[0]
        self.n_totSample = self.n_sample_c1 + self.n_sample_c2
        self.mu_c1 = np.mean(xs_class_1, axis=0)
        self.mu_c2 = np.mean(xs_class_2, axis=0)
        devia_c1 = np.dot( (xs_class_1-self.mu_c1).T , (xs_class_1-self.mu_c1) ) / self.n_sample_c1
        devia_c2 = np.dot( (xs_class_2-self.mu_c2).T , (xs_class_2-self.mu_c2) ) / self.n_sample_c2
        self.share_devia = (self.n_sample_c1/self.n_totSample)*devia_c1 + (self.n_sample_c2/self.n_totSample)*devia_c2

        self.deviaInv = np.linalg.inv(self.share_devia)
        self.w = np.dot((self.mu_c1-self.mu_c2), self.deviaInv)
        self.w = numpilize(self.w)
        self.b = (-0.5) * np.dot(np.dot(self.mu_c1, self.deviaInv), self.mu_c1.T) \
            + (-0.5) * np.dot(np.dot(self.mu_c2, self.deviaInv), self.mu_c2.T) \
            + np.log(float(self.n_sample_c1)/self.n_sample_c2)


    def ProbabilityIsClass_1(self, xs):
        xs = numpilize(xs)
        z = np.dot(xs, self.w.T) + self.b
        ans = self.Sigmoid(z)
        ans = np.clip(ans, 1e-11, 0.99999999999)
        return ans


    def ProbabilityIsClass_2(self, xs):
        return (1 - self.ProbabilityIsClass_1(xs))


    def Sigmoid(self, z):
        denom = 1 + np.exp(-z)
        ans = 1 / denom
        return ans










# x1 = np.random.randint(1,100,60).reshape(20,3)
# x2 = np.random.randint(1,100,15).reshape(5,3)
#
# pgm = PGMclassifier(x1,x2)
#
# print(pgm.ProbabilityIsClass_1(x1))