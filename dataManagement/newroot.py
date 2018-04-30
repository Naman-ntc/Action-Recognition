import numpy as np 
import pickle
import torch

def f(a):
	##array is 3d
	##size is 300x25x3
	#print(a.shape)
	first = 0
	last = 300
	zeros = np.zeros((75, 1))
	if not (a[299, :]==0).all():
		return a
	while (first<last):
		middle = (first + last)//2
		if (a[middle,:] == 0).all():
			last = middle
		else:
			first = middle + 1
	firstZeroIndex = min(first, last)
	return a[:firstZeroIndex]


trainData = pickle.load(open('../datasets/toyData/trainData.npy','rb'))
valData = pickle.load(open('../datasets/toyData/valData.npy','rb'))

trainLen = len(trainData)
valLen = len(valData)



for i in range(trainLen):
	thisData = trainData[i].numpy()
	thisData = thisData.reshape((-1,16,3))
	numFrames = thisData.shape[0]
	divisor = None
	subtractor = None
	for j in range(numFrames):
		if j==0:
			print(j)
			divisor = np.linalg.norm(thisData[j,6,:]-thisData[j,7,:])
			subtractor = thisData[j,6,:] - 0
		thisData[j,:,:] = thisData[j,:,:] - subtractor
		thisData[j] = thisData[j]/divisor
	trainData[i] = torch.from_numpy(thisData.reshape(-1,48))
	print("Training Video %d root relatived", i)


for i in range(valLen):
	thisData = valData[i].numpy()
	thisData = thisData.reshape((-1,16,3))
	numFrames = thisData.shape[0]
	divisor = None
	subtractor = None
	for j in range(numFrames):
		if j==0:
			print(j)
			divisor = np.linalg.norm(thisData[j,6,:]-thisData[j,7,:])
			subtractor = thisData[j,6,:] - 0
		thisData[j,:,:] = thisData[j,:,:] - subtractor
		thisData[j] = thisData[j]/divisor
	valData[i] = torch.from_numpy(thisData.reshape(-1,48))
	print("Validation Video %d root relatived", i)


pickle.dump(trainData,open('../datasets/trainData.npy','wb'))
pickle.dump(valData,open('../datasets/valData.npy','wb'))


