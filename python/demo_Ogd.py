import numpy
import scipy
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd

X_train,Y_train=load_svmlight_file("../data/a9a")
X_test,Y_test=load_svmlight_file("../data/a9a.t")

clf=Ogd(eta=0.1,power_t=0.5)
train_accuracy,data,err,fit_time=clf.fit(X_train,Y_train,True)

print("training accuracy:")
print(train_accuracy)

print("training time")
print(fit_time[-1])

print("Model sparsity")
print(clf.sparsity)

print("Instance prediction")
print(clf.predict(X_test))

print("Model weight")
print(clf.coef_)

print("Predict confidence scores for samples in X")
print(clf.decision_function(X_test))


test_accuracy,auc=clf.score(X_test,Y_test,True)
print("Test accuracy")
print(test_accuracy)
print("AUC")
print(auc)

clf.save("ogd_model_pylibol")

"""
print("along the learning process")
clf_2=Ogd(eta=0.1,power_t=0.5)
print(clf_2.fit(X_train,Y_train))
print(clf_2.fit(X_train,Y_train))
print(clf_2.fit(X_train,Y_train))
print(clf_2.fit(X_train,Y_train))
"""


