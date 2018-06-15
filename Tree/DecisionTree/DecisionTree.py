from sklearn.datasets import *;
from sklearn.tree import *;
from sklearn.model_selection import train_test_split;
import numpy as np;
import matplotlib.pyplot as plt;





iris = load_iris();
print("iris keys: ", iris.keys());
print("iris feature names: ", iris.feature_names);
print("iris target names: ", iris.target_names);
print("iris description: ",iris.DESCR);
samples = iris.data; # feature_1 = sample[:,0], feature_2 = sample[:,1], ...
outputs = iris.target;
NbrSamples, NbrFeatures = iris.data.shape;


plt.figure(0);
plt.scatter(samples[:,0],samples[:,1],c=outputs,marker='o');
formatter = plt.FuncFormatter(lambda i,*args: iris.target_names[int(i)]);
# iris.target has 3 type values : 0, 1, 2, so ticks=[0,1,2], each type value will apply to
# lambda 'i'.
plt.colorbar(ticks=[0,1,2], format=formatter);
plt.show();


X_train,X_test,Y_train,Y_test = train_test_split(samples,outputs,test_size=0.1);

DTclf = DecisionTreeClassifier(random_state=0);
DTclf.fit(X_train,Y_train);
predRes = DTclf.predict(X_test);
print("predict = ", predRes);
print("real = ", Y_test);

acc = np.sum(predRes == Y_test)/float(len(Y_test));
print("acc = ", acc);