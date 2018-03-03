import numpy as np 
from helperFunctions import *

def f(a, axis):
	##array is 3d
	##size is 300x25x3
	first = 0
	last = 300
	zeros = np.zeros((25, 3))
	while (first<last):
		middle = (first + last)//2
		if (a[middle,:,:] == 0).all():
			last = middle
		else:
			first = middle + 1

trainData = getData()

np.apply_along_axis(f, trainData, [1,2,3])