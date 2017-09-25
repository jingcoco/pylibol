import pysol
import numpy
import matplotlib.pyplot as plt

class Algorithm():
	def __init__(self, **params):
		self.params = params

	def fit(self,X,Y,showDraw):
		pass
		
	def score(self,X,Y,showDraw):
		pass


	def predict(self, X):
		return self.classifier.predict(X)

	def decision_function(self, X, Y=None, get_label=False):
		Y_scores = self.classifier.decision_function(X, Y, get_label)
    		return Y_scores

	@property
	def coef_(self, cls_id=0):
		return self.classifier.get_weight(cls_id)

	@property
	def sparsity(self):
		return self.classifier.sparsity

	def save(self, model_path):
		return self.classifier.save(model_path)
		
	def drawFit(self,X1_axis,Y1_axis,X1_name,Y1_name,X2_axis,Y2_axis,X2_name,Y2_name,X3_axis,Y3_axis,X3_name,Y3_name,algo_name):
	
		plt.figure("Error Rate")
		plt.plot(X1_axis,Y1_axis,"ro")
		plt.plot(X1_axis,Y1_axis,label=algo_name,color="red",linewidth=2)
		plt.xlabel(X1_name)
		plt.ylabel(Y1_name)
		plt.legend()
		
		plt.figure("Num of SV")
		plt.plot(X2_axis,Y2_axis,"bo")
		plt.plot(X2_axis,Y2_axis,label=algo_name,color="blue",linewidth=2)
		plt.xlabel(X2_name)
		plt.ylabel(Y2_name)
		plt.legend()
		
		plt.figure("Time")
		plt.plot(X3_axis,Y3_axis,"go")
		plt.plot(X3_axis,Y3_axis,label=algo_name,color="green",linewidth=2)
		plt.xlabel(X3_name)
		plt.ylabel(Y3_name)
		plt.legend()
		
		plt.show()
		
	def drawScore(self,X1_axis,Y1_axis,X1_name,Y1_name,X2_axis,Y2_axis,X2_name,Y2_name,algo_name):
		plt.figure("TPR VS FPR")
		plt.semilogx(X1_axis,Y1_axis,"r*")
		plt.semilogx(X1_axis,Y1_axis,label=algo_name,color="red",linewidth=2)
		plt.xlabel(X1_name)
		plt.ylabel(Y1_name)
		plt.legend()
		plt.show()
		
		print(
"\n\
+ - - - - - - + - - - - - - +\n\
      FPR           TPR   \n\
     "+ str(X2_axis[0]) +"       "+ str(Y2_axis[0]) +"   \n\
     "+ str(X2_axis[1]) +"      "+ str(Y2_axis[1]) +"   \n\
     "+ str(X2_axis[2]) +"       "+ str(Y2_axis[2]) +"   \n\
     "+ str(X2_axis[3]) +"        "+ str(Y2_axis[3]) +"   \n\
     "+ str(X2_axis[4]) +"         "+ str(Y2_axis[4]) +"   \n\
     "+ str(X2_axis[5]) +"         "+ str(Y2_axis[5]) +"   \n\
+ - - - - - - + - - - - - - +\n\
")
		
	

class OGD(Algorithm):

	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","OGD")
				
			return train_accuracy,self.data,self.err,self.time
		except AttributeError as e:
			self.classifier=pysol.SOL('OGD',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","OGD")
			return self.train_accuracy,self.data,self.err,self.time
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","OGD")
		
		return accuracy,auc,tpr_fig,fpr_fig

class PA(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA")	
			return self.train_accuracy,self.data,self.err,self.time
		except AttributeError as e:
			self.classifier=pysol.SOL('PA',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA")
			return self.train_accuracy,self.data,self.err,self.time
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","PA")
		
		return accuracy,auc,tpr_fig,fpr_fig


		
class AROW(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","AROW")	
			return self.train_accuracy,self.data,self.err,self.time
		except AttributeError as e:
			self.classifier=pysol.SOL('AROW',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","AROW")
			return self.train_accuracy,self.data,self.err,self.time
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","AROW")
		
		return accuracy,auc,tpr_fig,fpr_fig



class CW(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","CW")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('CW',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","CW")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","CW")
		
		return accuracy,auc,tpr_fig,fpr_fig

		
	
class PA1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA1")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('PA1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","PA1")
		
		return accuracy,auc,tpr_fig,fpr_fig




class PA2(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA2")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('PA2',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PA2")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","PA2")
		
		return accuracy,auc,tpr_fig,fpr_fig



class Ada_FOBOS_L1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_FOBOS_L1")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada-FOBOS-L1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_FOBOS_L1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_FOBOS_L1")
		
		return accuracy,auc,tpr_fig,fpr_fig



class Ada_FOBOS(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_FOBOS")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada-fobos',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_FOBOS")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_FOBOS")
		
		return accuracy,auc,tpr_fig,fpr_fig



class Ada_RDA(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_RDA")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada-rda',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_RDA")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_RDA")
		
		return accuracy,auc,tpr_fig,fpr_fig



class Ada_RDA_L1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_RDA_L1")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada-rda-l1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ada_RDA_L1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_RDA_L1")
		
		return accuracy,auc,tpr_fig,fpr_fig


class ALMA2(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ALMA2")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Alma2',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ALMA2")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","ALMA2")
		
		return accuracy,auc,tpr_fig,fpr_fig

		


class ECCW(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ECCW")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Eccw',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ECCW")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","ECCW")
		
		return accuracy,auc,tpr_fig,fpr_fig



class ERDA_L1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ERDA_L1")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Erda-l1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","ERDA_L1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","ERDA_L1")
		
		return accuracy,auc,tpr_fig,fpr_fig



class FOBOS_L1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","FOBOS_L1")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Fobos-l1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","FOBOS_L1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","FOBOS_L1")
		
		return accuracy,auc,tpr_fig,fpr_fig



class FOFS(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","FOFS")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Fofs',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","FOFS")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","FOFS")
		
		return accuracy,auc,tpr_fig,fpr_fig



class Perceptron(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Perceptron")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Perceptron',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Perceptron")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Perceptron")
		
		return accuracy,auc,tpr_fig,fpr_fig



class PET(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PET")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Pet',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","PET")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","PET")
		
		return accuracy,auc,tpr_fig,fpr_fig



class RDA(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","RDA")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Rda',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","RDA")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","RDA")
		
		return accuracy,auc,tpr_fig,fpr_fig



class RDA_L1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","RDA_L1")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Rda-l1',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","RDA_L1")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","RDA_L1")
		
		return accuracy,auc,tpr_fig,fpr_fig



class SOFS(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","SOFS")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Sofs',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","SOFS")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","SOFS")
		
		return accuracy,auc,tpr_fig,fpr_fig





class SOP(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","SOP")	
			return self.train_accuracy,self.data,self.err,self.time 
		except AttributeError as e:
			self.classifier=pysol.SOL('Sop',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","SOP")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","SOP")
		
		return accuracy,auc,tpr_fig,fpr_fig


		
class STG(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y,showDraw=False):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
		
			self.update=numpy.concatenate((self.update,update))
			self.data=numpy.concatenate((self.data,data))
			self.iter=numpy.concatenate((self.iter,iter))
			self.err=numpy.concatenate((self.err,err))
			time=time+self.time[-1]
			self.time=numpy.concatenate((self.time,time))
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","STG")	
			return self.train_accuracy,self.data,self.err,self.time  
		except AttributeError as e:
			self.classifier=pysol.SOL('Stg',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","STG")
			return self.train_accuracy,self.data,self.err,self.time 
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","STG")
		
		return accuracy,auc,tpr_fig,fpr_fig
