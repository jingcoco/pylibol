import numpy
import scipy
from sklearn.datasets import load_svmlight_file
from sklearn import datasets
from classifiers import Alma2
import time

iris=datasets.load_iris()
X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")

start_time = time.time()
clf=Alma2()
train_accuracy=clf.fit(X_train,Y_train)
print("training accuracy:")
print(train_accuracy)
train_time = time.time() - start_time
print("training time cost")
print(train_time)

print("sparsity")
print(clf.sparsity)

print("prediction of instance")
y=clf.predict(X_test)
print(y)

print("model weight")
print(clf.coef_)

clf.save("Alma2_model")