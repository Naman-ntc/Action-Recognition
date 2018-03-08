import numpy as np 
from helperFunctions import *

def f(a):
	##array is 3d
	##size is 300x25x3
	print(a.shape)
	first = 0
	last = 300
	zeros = np.zeros((25, 3))
	if not (a[299, :, :]==0).all():
		return a
	while (first<last):
		middle = (first + last)//2
		if (a[middle,:,:] == 0).all():
			last = middle
		else:
			first = middle + 1
	firstZeroIndex = min(first, last)
	currentIndex = firstZeroIndex
	while currentIndex + firstZeroIndex < 300:
		a[currentIndex:currentIndex+firstZeroIndex,:,:] = a[:firstZeroIndex,:,:]
		currentIndex += firstZeroIndex
	howMuch = 300 - currentIndex
	a[currentIndex:] = a[:howMuch]
	return a

trainData = getData()

for i in range(trainData.shape[0]):
	trainData[i,:,:,:] = f(trainData[i,:,:,:])

np.save(open("Final-Data2/train_data.npy", 'wb'), trainData)