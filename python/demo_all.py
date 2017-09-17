import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")

print("Ogd")
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
y=clf.predict(X_test[:10])
print(y)

print("model weight")
print(clf.coef_)

print("decision function")
print(clf.decision_function(X_test))

clf.save("Ogd_model")