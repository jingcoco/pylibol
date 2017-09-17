import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")

start_time = time.time()
clf=Ogd(eta=0.1,power_t=0.5)
train_accuracy=clf.fit(X_train,Y_train)
print("training accuracy:")
print(train_accuracy)
train_time = time.time() - start_time
print("training time")
print(train_time)

print("model sparsity")
print(clf.sparsity)

print("instance prediction")
y=clf.predict(X_test)
print(y)

print("model weight")
print(clf.coef_)

print("decision function")
print(clf.decision_function(X_test))

clf.save("ogd_model_pylibol")

print("along the learning process")
clf_2=Ogd(eta=0.1,power_t=0.5)
print(clf_2.fit(X_train[0:100],Y_train[0:100]))
print(clf_2.fit(X_train[100:200],Y_train[100:200]))
print(clf_2.fit(X_train[200:300],Y_train[200:300]))
print(clf_2.fit(X_train[300:400],Y_train[300:400]))






