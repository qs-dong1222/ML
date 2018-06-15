import os
import numpy as xp
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


# 60% features will be randomly bootstrapped from feature set
# 500 decision trees
# 8 threads in parallel
# enable bootstrap
RF_model = RandomForestClassifier(n_estimators=500,
                                  max_features=0.6,
                                  n_jobs=8,
                                  bootstrap=True).fit(xs_train, ys_train)


print("feature NO.", xp.argmax(RF_model.feature_importances_), " is the most important feature")


acc = RF_model.score(xs_test, ys_test)
print("test set accuracy =", acc)


pred_cls = RF_model.predict( [xs_test[0]] )
print("class prediction of given data sample:\n", pred_cls)
pred_cls_prob = RF_model.predict_proba( [xs_test[0]])
print("probability of being each class of given data sample:\n", pred_cls_prob)



# clean up
if(os.path.exists("features.npy")):
    os.remove("features.npy")

if(os.path.exists("labels.npy")):
    os.remove("labels.npy")