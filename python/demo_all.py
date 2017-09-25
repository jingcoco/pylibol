import numpy
import scipy
from sklearn.datasets import load_svmlight_file
from classifiers import Ada_fobos_l1,Ada_fobos,Ada_rda,Ada_rda_l1,Erda_l1,Fobos_l1,Rda_l1

X_train,Y_train=load_svmlight_file("../data/a7a")
X_test,Y_test=load_svmlight_file("../data/a7a.t")

clf=Ada_rda_l1()
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





