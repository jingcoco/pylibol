import pysol
import target_process
import numpy
import matplotlib.pyplot as plt

class Algorithm():
	def __init__(self, **params):
		self.params = params

	def fit():
		pass

	def predict(self, X):
		Y_predict = self.classifier.predict(X)
		Y_original = numpy.empty(shape=[0, len(Y_predict)])
		for i in range(len(Y_predict)):
			Y_original = numpy.append(Y_original, self.new_to_old[Y_predict[i]])
		return Y_original

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
		plt.plot(X1_axis,Y1_axis,"r*")
		plt.plot(X1_axis,Y1_axis,label=algo_name,color="red",linewidth=2)
		plt.xlabel(X1_name)
		plt.ylabel(Y1_name)
		plt.legend()
		
		plt.figure("Num of SV")
		plt.plot(X2_axis,Y2_axis,"r*")
		plt.plot(X2_axis,Y2_axis,label=algo_name,color="blue",linewidth=2)
		plt.xlabel(X2_name)
		plt.ylabel(Y2_name)
		plt.legend()
		
		plt.figure("Time")
		plt.plot(X3_axis,Y3_axis,"r*")
		plt.plot(X3_axis,Y3_axis,label=algo_name,color="green",linewidth=2)
		plt.xlabel(X3_name)
		plt.ylabel(Y3_name)
		plt.legend()
		
		plt.show()
		
	def drawScore(self,X1_axis,Y1_axis,X1_name,Y1_name,X2_axis,Y2_axis,X2_name,Y2_name,algo_name):
		plt.figure("TPR VS FPR")
		plt.plot(X1_axis,Y1_axis,"r*")
		plt.plot(X1_axis,Y1_axis,label=algo_name,color="red",linewidth=2)
		plt.xlabel(X1_name)
		plt.ylabel(Y1_name)
		plt.legend()
		
		plt.figure("Table of TPR VS FPR")
		plt.plot(X2_axis,Y2_axis,"r*")
		plt.plot(X2_axis,Y2_axis,label=algo_name,color="blue",linewidth=2)
		plt.xlabel(X2_name)
		plt.ylabel(Y2_name)
		plt.legend()
		
		plt.show()
		
	def score(self,X,Y):
		pass


class Ogd(Algorithm):

	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		[Y_new,self.old_to_new,self.new_to_old,num_class]=target_process.translate(Y)
		try: 
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y_new)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ogd")
			return train_accuracy		    
		except AttributeError as e:
			self.classifier=pysol.SOL('ogd',num_class,**self.params)
			train_accuracy,update,data,iter,err,time = self.classifier.fit(X,Y_new)
			self.drawFit(data,err,"data","error",data,update,"data","update",data,time,"data","time","Ogd")
			return train_accuracy
			
	def score(self,X,Y):
		[Y_new,self.old_to_new,self.new_to_old,num_class]=target_process.translate(Y)
		
		accuracy,tpr_fig,fpr_fig,tpr_tab,fpr_tab,auc = self.classifier.score(X,Y_new)
		self.drawScore(tpr_fig,fpr_fig,"tpr","fpr",tpr_tab,fpr_tab,"tpr","fpr","Ogd")
		return accuracy,auc

class Pa(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('pa', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy




class Pa1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('pa1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy




class Pa2(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('pa2', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Ada_fobos_l1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('ada-fobos-l1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Ada_fobos(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('ada-fobos', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Ada_rda(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('ada-rda', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy




class Ada_rda_l1(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('ada-rda-l1', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy


class Alma2(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('alma2', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Arow(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('arow', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Cw(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('cw', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



class Eccw(Algorithm):
	def __init__(self, **params):
		self.params = params

	def fit(self, X, Y):
		[Y_new, self.old_to_new, self.new_to_old, num_class] = target_process.translate(Y)
		try:
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy
		except AttributeError as e:
			self.classifier = pysol.SOL('eccw', num_class, **self.params)
			train_accuracy = self.classifier.fit(X, Y_new)
			return train_accuracy



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