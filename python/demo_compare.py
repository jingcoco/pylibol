import numpy
import scipy
import time
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
from classifiers import OGD,PA,PA1,PA2,AROW,CW,Ada_FOBOS,Ada_FOBOS_L1,FOBOS_L1,Ada_RDA,Ada_RDA_L1,ALMA2,ECCW,ERDA_L1,FOFS,Perceptron,PET,RDA,RDA_L1,SOFS,SOP,STG

X_train,Y_train=load_svmlight_file("../data/a9a")
X_test,Y_test=load_svmlight_file("../data/a9a.t")


clf=OGD()
OGD_train_accuracy,OGD_data,OGD_error,OGD_time=clf.fit(X_train,Y_train)
accuracy,auc,OGD_tpr_fig,OGD_fpr_fig=clf.score(X_test,Y_test)

clf2=PA()
PA_train_accuracy,PA_data,PA_error,PA_time=clf2.fit(X_train,Y_train)
accuracy,auc,PA_tpr_fig,PA_fpr_fig= clf2.score(X_test,Y_test)

clf3=PA1()
PA1_train_accuracy,PA1_data,PA1_error,PA1_time=clf3.fit(X_train,Y_train)
accuracy,auc,PA1_tpr_fig,PA1_fpr_fig = clf3.score(X_test,Y_test)

clf4=PA2()
PA2_train_accuracy,PA2_data,PA2_error,PA2_time=clf4.fit(X_train,Y_train)
accuracy,auc,PA2_tpr_fig,PA2_fpr_fig = clf4.score(X_test,Y_test)

clf5=AROW()
AROW_train_accuracy,AROW_data,AROW_error,AROW_time=clf5.fit(X_train,Y_train)
accuracy,auc,AROW_tpr_fig,AROW_fpr_fig= clf5.score(X_test,Y_test)

clf6=CW()
CW_train_accuracy,CW_data,CW_error,CW_time=clf6.fit(X_train,Y_train)
accuracy,auc,CW_tpr_fig,CW_fpr_fig= clf6.score(X_test,Y_test)

clf7=Ada_FOBOS()
Ada_FOBOS_train_accuracy,Ada_FOBOS_data,Ada_FOBOS_error,Ada_FOBOS_time=clf7.fit(X_train,Y_train)
accuracy,auc,Ada_FOBOS_tpr_fig,Ada_FOBOS_fpr_fig= clf7.score(X_test,Y_test)

clf8=Ada_RDA()
Ada_RDA_train_accuracy,Ada_RDA_data,Ada_RDA_error,Ada_RDA_time=clf8.fit(X_train,Y_train)
accuracy,auc,Ada_RDA_tpr_fig,Ada_RDA_fpr_fig= clf8.score(X_test,Y_test)

clf9=ALMA2()
ALMA2_train_accuracy,ALMA2_data,ALMA2_error,ALMA2_time=clf9.fit(X_train,Y_train)
accuracy,auc,ALMA2_tpr_fig,ALMA2_fpr_fig= clf9.score(X_test,Y_test)

clf10=ECCW()
ECCW_train_accuracy,ECCW_data,ECCW_error,ECCW_time=clf10.fit(X_train,Y_train)
accuracy,auc,ECCW_tpr_fig,ECCW_fpr_fig= clf10.score(X_test,Y_test)


"""Draw figure: Compare error
"""
plt.figure("Compare error")
plt.plot(OGD_data,OGD_error,"ro-",label="OGD",markersize=5,linewidth=2)
plt.plot(PA_data,PA_error,"bo-",label="PA",markersize=5)
plt.plot(PA1_data,PA1_error,"go-",label="PA1",markersize=5)
plt.plot(PA2_data,PA2_error,"ko-",label="PA2",markersize=5)
plt.plot(AROW_data,AROW_error,"mo-",label="AROW",markersize=5)
plt.plot(CW_data,CW_error,"yo-",label="CW",markersize=5)
plt.plot(Ada_FOBOS_data,Ada_FOBOS_error,"co-",label="Ada_FOBOS",markersize=5)
plt.plot(Ada_RDA_data,Ada_RDA_error,"r*-",label="Ada_FOBOS_data",markersize=5)
plt.plot(ALMA2_data,ALMA2_error,"b*-",label="ALMA2",markersize=5)
plt.plot(ECCW_data,ECCW_error,"g*-",label="ECCW",markersize=5)

plt.grid(True)
plt.xlabel("data")
plt.ylabel("error rate")
plt.legend(loc="upper right")


"""Draw figure: Compare time
"""
plt.figure("Compare time")
plt.plot(OGD_data,OGD_time,"ro-",label="OGD",markersize=5)
plt.plot(PA_data,PA_time,"bo-",label="PA",markersize=5)
plt.plot(PA1_data,PA1_time,"go-",label="PA1",markersize=5)
plt.plot(PA2_data,PA2_time,"ko-",label="PA2",markersize=5)
plt.plot(AROW_data,AROW_time,"mo-",label="Arow",markersize=5)
plt.plot(CW_data,CW_time,"yo-",label="CW",markersize=5)
plt.plot(Ada_FOBOS_data,Ada_FOBOS_time,"co-",label="Ada_FOBOS",markersize=5)
plt.plot(Ada_RDA_data,Ada_RDA_time,"r*-",label="Ada_FOBOS_data",markersize=5)
plt.plot(ALMA2_data,ALMA2_time,"b*-",label="ALMA2",markersize=5)
plt.plot(ECCW_data,ECCW_time,"g*-",label="ECCW",markersize=5)

plt.grid(True)
plt.xlabel("data")
plt.ylabel("time")
plt.legend()


"""Draw figure: Compare Roc
"""
plt.figure("FPR VS TPR")
plt.plot(OGD_fpr_fig,OGD_tpr_fig,"ro-",label="OGD",markersize=5)
plt.plot(PA_fpr_fig,PA_tpr_fig,"bo-",label="PA",markersize=5)
plt.plot(PA1_fpr_fig,PA1_tpr_fig,"go-",label="PA1",markersize=5)
plt.plot(PA2_fpr_fig,PA2_tpr_fig,"ko-",label="PA2",markersize=5)
plt.plot(AROW_fpr_fig,AROW_tpr_fig,"mo-",label="Arow",markersize=5)
plt.plot(CW_fpr_fig,CW_tpr_fig,"yo-",label="CW",markersize=5)
plt.plot(Ada_FOBOS_fpr_fig,Ada_FOBOS_tpr_fig,"co-",label="Ada_FOBOS",markersize=5)
plt.plot(Ada_RDA_fpr_fig,Ada_RDA_tpr_fig,"r*-",label="Ada_FOBOS_data",markersize=5)
plt.plot(ALMA2_fpr_fig,ALMA2_tpr_fig,"b*-",label="ALMA2",markersize=5)
plt.plot(ECCW_fpr_fig,ECCW_tpr_fig,"g*-",label="ECCW",markersize=5)

plt.grid(True)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.show()

