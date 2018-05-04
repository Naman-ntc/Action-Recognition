import numpy as np

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

data = np.zeros((5, 300, 25, 3))
data[0,:25,:,:] = np.arange(25*25*3).reshape(25, 25, 3)
data[1,:32,:,:] = np.arange(32*25*3).reshape(32, 25, 3)
data[2,:,:,:] = np.arange(300*25*3).reshape(300, 25, 3)
data[3,:12,:,:] = np.arange(12*25*3).reshape(12, 25, 3)
data[4,:20,:,:] = np.arange(20*25*3).reshape(20, 25, 3)


#ans = np.apply_over_axes(f, data, [1,2,3])

for i in range(data.shape[0]):
	data[i,:,:,:] = f(data[i,:,:,:])

print(data[0,0,:,:])
print(data.shape)