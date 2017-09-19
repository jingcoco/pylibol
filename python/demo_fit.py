import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")

print("along the learning process")
clf=Ogd(eta=0.1,power_t=0.5)

train_accuracy,data,err,fit_time=clf.fit(X_train,Y_train,True)
print(train_accuracy)
train_accuracy,data,err,fit_time=clf.fit(X_train,Y_train,True)
print(train_accuracy)
train_accuracy,data,err,fit_time=clf.fit(X_train,Y_train,True)
print(train_accuracy)
