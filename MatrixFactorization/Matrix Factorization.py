"""
The idea of Matrixfactorization is treating the given matrix as the multiplication result
between 2 matrice. THe element in these 2 matrice are features waiting to be trained.
One of the 2 matrix represents the feature that each row has such that the row is able to
approach to the multiplication result. Same thing for the other matrix but represents the
features for each column. The number of features are randomly defined which correspondes 
to the "feature_dim" parameter below.
"""

from Batch import *



class MatrixFactorizer:
    """
    typical input format:
    Y_true = [[5, 4, 1],
              [5, 5, 0],
              [np.NaN, 4, 1],
              [1, np.NaN, 5],
              [0, 0, 5]]

    feature_dim: hidden number of features(parameters) that we want to assign to
                 each row/column catagory. Theoretically, the higher this value is,
                 the more precision you can get.
    """
    def __init__(self, Y_true, feature_dim):
        self.n_features = round(feature_dim)
        self.Y_mask = np.isnan(Y_true)
        self.Y_true = np.ma.array(numpilize(Y_true), mask=self.Y_mask)

        self.X = np.random.rand(self.Y_true.shape[0], self.n_features)
        self.W = np.random.rand(self.Y_true.shape[1], self.n_features)
        self.b = np.ones((1, self.Y_true.shape[1]))


    #############################
    # predict function
    # withMask: tells if we predict the 'NaN' positions
    #############################
    def predict(self, withMask=False):
        mat_b = np.tile(self.b, (self.Y_true.shape[0], 1))
        Y_pred = np.dot(self.X, self.W.T) + mat_b
        if(withMask):
            Y_pred = np.ma.array(Y_pred, mask=self.Y_mask)
        return Y_pred




    #####################
    # loss function
    #####################
    def loss(self):
        Y_pred = self.predict(withMask=True)
        lossVal = 0.5 * np.sum(np.square(Y_pred - self.Y_true)) \
               + 0.5 * self.regArg * np.sum(np.square(self.X)) \
               + 0.5 * self.regArg * np.sum(np.square(self.W))

        return lossVal



    #####################
    # gradient function
    # gradMode: this argument can be 'wb', 'xb' or whatever else
    #           'wb' means calculate the gradient on weights and bias
    #           'xb' means calculate the gradient on features and bias
    #           whatever else means calculate the gradient on weights, features and bias
    #####################
    def gradient(self, gradMode):
        Y_pred = self.predict(withMask=True)
        error = Y_pred - self.Y_true

        G_w = np.zeros(self.W.shape)
        G_x = np.zeros(self.X.shape)
        G_b = []

        if(gradMode=='wb'):
            for col in range(error.shape[1]):
                err_col_r = error[:, col]
                err_col = err_col_r[:, np.newaxis]

                grad_w = np.sum(self.X * err_col, axis=0)
                G_w[col] = grad_w

                grad_b = np.sum(err_col, axis=0)
                G_b.append(grad_b[0])
            G_b = numpilize(G_b)
            return G_w, G_b

        elif(gradMode=='xb'):
            for col in range(error.shape[1]):
                err_col_r = error[:, col]
                err_col = err_col_r[:, np.newaxis]

                grad_b = np.sum(err_col, axis=0)
                G_b.append(grad_b[0])

                err_col_rep = np.tile(err_col, (1, self.n_features))
                grad_x = err_col_rep * self.W[col]
                G_x += grad_x
                #print("grad_x: ", grad_x) ######################
            G_x /= error.shape[1]
            G_b = numpilize(G_b)
            return G_x, G_b

        else:
            for col in range(error.shape[1]):
                err_col_r = error[:, col]
                err_col = err_col_r[:, np.newaxis]

                grad_w = np.sum(self.X * err_col, axis=0)
                G_w[col] = grad_w

                grad_b = np.sum(err_col, axis=0)
                G_b.append(grad_b[0])

                err_col_rep = np.tile(err_col, (1, self.n_features))
                grad_x = err_col_rep * self.W[col]
                G_x += grad_x
            G_x /= error.shape[1]
            G_b = numpilize(G_b)
            return G_w, G_x, G_b



    #####################
    # update function
    # lr: learning rate
    # regArg: regularization lambda
    # updateMode: this argument can be 'wb', 'xb' or whatever else
    #             'wb' means update weights and bias
    #             'xb' means update features and bias
    #             whatever else means update weights, features and bias
    #####################
    def updateParam(self, lr=0.01, regArg=0.1, updateMode=None):
        if(updateMode=='wb'):
            G_w, G_b = self.gradient(updateMode)
            self.W = (1 - lr * regArg) * self.W - lr * G_w
            self.b = self.b - lr * G_b

        elif(updateMode=='xb'):
            G_x, G_b = self.gradient(updateMode)
            self.X = (1 - lr * regArg) * self.X - lr * G_x
            self.b = self.b - lr * G_b

        else:
            G_w, G_x, G_b = self.gradient()
            self.W = (1 - lr * regArg) * self.W - lr * G_w
            self.X = (1 - lr * regArg) * self.X - lr * G_x
            self.b = self.b - lr * G_b



    #####################
    # fit function
    # lr: learning rate
    # regArg: regularization lambda
    #####################
    def fit(self, lr=0.01, regArg=0.1, train_times=1000, showLoss=False):
        self.regArg = regArg
        for i in range(2*train_times):
            if(i%2==0):
                self.updateParam(lr=lr, regArg=self.regArg, updateMode='wb')
            else:
                self.updateParam(lr=lr, regArg=self.regArg, updateMode='xb')

            if(i%50==0 and showLoss):
                print(self.loss())














Y_true = [[10,9,15],
         [28,7,6],
         [np.NaN,9,1],
         [1,np.NaN,5],
         [0,0,5]]


mf = MatrixFactorizer(Y_true, 6)

pred = mf.predict()
print(pred)

mf.fit(lr=0.01, regArg=0, train_times=2000, showLoss=True)

pred = mf.predict()
print(pred)
