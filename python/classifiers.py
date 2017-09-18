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
		
	

class Ogd(Algorithm):

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
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ogd")
				
			return train_accuracy 
		except AttributeError as e:
			self.classifier=pysol.SOL('ogd',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Ogd")
			return self.train_accuracy
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ogd")
		
		return accuracy,auc

class Pa(Algorithm):
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
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Pa")	
			return train_accuracy 
		except AttributeError as e:
			self.classifier=pysol.SOL('Pa',num_class,**self.params)
			self.train_accuracy,self.update,self.data,self.iter,self.err,self.time = self.classifier.fit(X,Y)
			
			if(showDraw == True):
				self.drawFit(self.data,self.err,"data","error",self.data,self.update,"data","update",self.data,self.time,"data","time","Pa")
			return self.train_accuracy
			
	def score(self,X,Y,showDraw=False):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		
		if(showDraw == True):
			self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Pa")
		
		return accuracy,auc



"""
class Pa1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Pa1")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Pa1',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Pa1")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Pa1")
		return accuracy,auc




class Pa2(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Pa2")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Pa2',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Pa2")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Pa2")
		return accuracy,auc



class Ada_fobos_l1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_fobos_l1")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada_fobos_l1',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_fobos_l1")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_fobos_l1")
		return accuracy,auc



class Ada_fobos(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_fobos")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada_fobos',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_fobos")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_fobos")
		return accuracy,auc



class Ada_rda(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_rda")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada_rda',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_rda")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_rda")
		return accuracy,auc





class Ada_rda_l1(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_rda_l1")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Ada_rda_l1',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ada_rda_l1")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Ada_rda_l1")
		return accuracy,auc


class Alma2(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Alma2")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Alma2',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Alma2")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Alma2")
		return accuracy,auc



class Arow(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Arow")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Arow',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Arow")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Arow")
		return accuracy,auc



class Cw(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Cw")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Cw',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Cw")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Cw")
		return accuracy,auc



class Eccw(Algorithm):
	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		num_class=len(set(Y))
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Eccw")
			return train_accuracy	    
		except AttributeError as e:
			self.classifier=pysol.SOL('Eccw',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Eccw")
			return train_accuracy
			
	def score(self,X,Y):
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y)
		self.drawScore(fpr_fig,tpr_fig,"fpr","tpr",fpr_tab,tpr_tab,"fpr","tpr","Eccw ")
		return accuracy,auc



class Erda_l1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('erda-l1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Fobos_l1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('fobos-l1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Fofs(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('fofs', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Perceptron(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('perceptron', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Pet(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('pet', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Rda(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('rda', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Rda_l1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('rda-l1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Sofs(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('sofs', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy





class Sop(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('sop', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Stg(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('stg', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
"""