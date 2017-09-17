import pysol
import target_process
import numpy

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



class Ogd(Algorithm):

	def __init__(self,**params):
		self.params=params

	def fit(self,X,Y):
		[Y_new,self.old_to_new,self.new_to_old,num_class]=target_process.translate(Y)
		try: 
			train_accuracy=self.classifier.fit(X,Y_new)
			return train_accuracy		    
		except AttributeError as e:
			self.classifier=pysol.SOL('ogd',num_class,**self.params)
			train_accuracy=self.classifier.fit(X,Y_new)
			return train_accuracy
	


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