import numpy

def translate(Y_original):
	target_set=set(Y_original)
	n_class=len(target_set)
	old_to_new=dict()
	new_to_old=dict()
		
	j=0
	
	if n_class==1:
		raise Exception("Only ONE class detected")
	if n_class==2:
		new=numpy.array([1,-1])
	if n_class>2:
		new=numpy.array(range(n_class))
		
	for i in target_set:
		old_to_new[i]=new[j]
		new_to_old[new[j]]=i
		j=j+1
		
	for i in range(len(Y_original)):
		Y_original[i]=old_to_new[Y_original[i]]
			
	return Y_original.astype(numpy.float64), old_to_new, new_to_old,n_class