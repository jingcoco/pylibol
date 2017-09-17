import numpy
import scipy
from sklearn.datasets import load_svmlight_file
from sol_classifiers import ogd
import time

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")

start_time = time.time()
clf=ogd()
print(clf.fit(X_train,Y_train))
print(clf.score(X_test,Y_test))






