"""
In sklearn, Gradient Boosting use only ID3 as estimator
"""

import numpy as xp
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import zero_one_loss
import matplotlib.pyplot as plt


xs = xp.load("features.npy")
ys_true = xp.load("labels.npy")

print("features.shape:\n", xs.shape)
print("labels.shape:\n", ys_true.shape)


# 2nd order polynomial to dataset
xs_poly = PolynomialFeatures(1).fit_transform(xs)

# min/max normalization to normalize feature to range (min, max)
xs_poly_norm = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(xs_poly)
# print("normed train set =\n", xs_normed)


xs_train, xs_test, ys_train, ys_test = train_test_split(xs_poly_norm, ys_true, test_size=0.33)


# 100 boosting stages
# 50% sub-bagging without replacement(无放回采样)
# 0.1 learning rate for contribution of week estimator in each boosting stage
n_estimators = 100
GBRT_model = GradientBoostingClassifier(n_estimators=n_estimators,
                                        subsample=0.5,
                                        learning_rate=0.1).fit(xs_train, ys_train)


print("feature NO.", xp.argmax(GBRT_model.feature_importances_), " is the most important feature")


acc = GBRT_model.score(xs_test, ys_test)
print("test set accuracy =", acc)


pred_cls = GBRT_model.predict( [xs_test[0]] )
print("class prediction of given data sample:\n", pred_cls)
pred_cls_prob = GBRT_model.predict_proba( xs_test[0:2] )
print("probability of being each class of given data sample:\n", pred_cls_prob)
print(pred_cls_prob[:,1][:,xp.newaxis])

stage_errs = []
# loop over all prediction of test set at every stage
for ys_pred in GBRT_model.staged_predict(xs_test):
    stage_err = zero_one_loss(y_pred=ys_pred, y_true=ys_test)
    stage_errs.append(stage_err)

plt.plot(stage_errs,
        label='GBRT Test Error based on ID3',
        color='blue')
plt.show()



# clean up
# if(os.path.exists("features.npy")):
#     os.remove("features.npy")
#
# if(os.path.exists("labels.npy")):
#     os.remove("labels.npy")


