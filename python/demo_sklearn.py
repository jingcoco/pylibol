import numpy
import scipy
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

iris=datasets.load_iris()
X_train=numpy.concatenate([iris.data[1:40],iris.data[50:90]])
Y_train=numpy.concatenate([iris.target[1:40],iris.target[50:90]])


X_test=numpy.concatenate([iris.data[40:50],iris.data[90:100]])
Y_test=numpy.concatenate([iris.target[40:50],iris.target[90:100]])


# ogd

print("ogd")
clf=SGDClassifier(loss='hinge',penalty="l2",max_iter=100)
clf.fit(X_train,Y_train)

print(clf.coef_)
print(clf.intercept_ )

print(clf.predict(X_test))

print("PA")

# PA

clf_PA=PassiveAggressiveClassifier()
clf_PA.fit(X_train,Y_train)

print(clf_PA.coef_)
print(clf_PA.intercept_ )

print(clf_PA.predict(X_test))
