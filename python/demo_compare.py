import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd,Pa,Arow
import matplotlib.pyplot as plt

X_train,Y_train=load_svmlight_file("../data/a1a.t")
X_test,Y_test=load_svmlight_file("../data/a1a")


start_time = time.time()
clf=Ogd(eta=0.1,power_t=0.5)
Pa_train_accuracy,Ogd_data,Ogd_error,Ogd_time=clf.fit(X_train,Y_train)

clf2=Pa()
Pa_train_accuracy,Pa_data,Pa_error,Pa_time=clf2.fit(X_train,Y_train)

clf3=Arow()
Arow_train_accuracy,Arow_data,Arow_error,Arow_time=clf3.fit(X_train,Y_train)

plt.figure("Compare error")
plt.plot(Ogd_data,Ogd_error,"ro")
plt.plot(Ogd_data,Ogd_error,label="Ogd",color="red",linewidth=2)
plt.plot(Pa_data,Pa_error,"b*")
plt.plot(Pa_data,Pa_error,label="Pa",color="blue",linewidth=2)
plt.plot(Arow_data,Arow_error,"g*")
plt.plot(Arow_data,Arow_error,label="Arow",color="green",linewidth=2)
plt.xlabel("data")
plt.ylabel("error rate")
plt.legend()

plt.figure("Compare time")
plt.plot(Ogd_data,Ogd_time,"ro")
plt.plot(Ogd_data,Ogd_time,label="Ogd",color="red",linewidth=2)
plt.plot(Pa_data,Pa_time,"b*")
plt.plot(Pa_data,Pa_time,label="Pa",color="blue",linewidth=2)
plt.plot(Arow_data,Arow_time,"g*")
plt.plot(Arow_data,Arow_time,label="Arow",color="green",linewidth=2)
plt.xlabel("data")
plt.ylabel("time")
plt.legend()

plt.show()

"""
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


test_accuracy,auc=clf.score(X_test,Y_test)
print("Test accuracy")
print(test_accuracy)
print("AUC")
print(auc)

clf.save("ogd_model_pylibol")
"""