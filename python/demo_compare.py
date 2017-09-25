import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
from classifiers import Ogd,Pa,Arow
import matplotlib.pyplot as plt

X_train,Y_train=load_svmlight_file("../data/a7a")
X_test,Y_test=load_svmlight_file("../data/a7a.t")



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

