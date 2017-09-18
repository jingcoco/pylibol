import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Pa

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")


start_time = time.time()
clf=Pa()
train_accuracy=clf.fit(X_train,Y_train,True)


print("training accuracy:")
print(train_accuracy)
train_time = time.time() - start_time
print("training time")
print(train_time)

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