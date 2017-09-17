import pysol
import target_process
import numpy
import matplotlib.pyplot as plt

class ogd:

	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			return self.classifier.fit(X,Y) 	    
		except AttributeError as e:
			self.classifier=pysol.SOL('ogd',num_class,**self.params)
			accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			plt.figure
			plt.plot(data,err,"r*")
			plt.plot(data,err,label="test",color="red",linewidth=2)
			plt.xlabel("data")
			plt.ylabel("error")
			plt.legend()
			plt.show()
			return accuracy
	@property
	def coef_(self, cls_id=0):
		return self.classifier.get_weight(cls_id)

	def predict(self, X):
		return self.classifier.predict(X)
	def score(self,X,Y):
		return self.classifier.score(X,Y)
