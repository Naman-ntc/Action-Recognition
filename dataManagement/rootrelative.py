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

#trainData = np.swapaxes(trainData, 1,2)
#trainData = np.swapaxes(trainData, 2,3)


"""
for i in range(trainData.shape[0]):
	for j in range(300):
		trainData[i,j] = trainData[i,j] - trainData[i,j,0]
		if ((trainData[i,j,1,:])**2).mean() != 0:
			trainData[i,j] = trainData[i,j]*(1.0/np.linalg.norm(trainData[i,j,1]))
	if i%50 == 0:
		print("Processing", i)

trainData = trainData.reshape(trainData.shape[0], 300, 75)
finalData = []

for i in range(trainData.shape[0]):
	finalData.append(torch.from_numpy(f(trainData[i])))

print("Processed!!!")

pickle.dump(finalData, open("../datasets/toyData/lstmProcessedValData.npy", 'wb'))
"""

trainLen = len(trainData)
valLen = len(valData)

for i in range(trainLen):
	thisData = trainData[i]
	thisData = thisData.reshape((-1,16,3))
	numFrames = thisData.shape[0]
	divisor = None
	for j in range(numFrames):
		thisData[j,:,:] = thisData[j,:,:] - this[0,6,:]
		if j==0:
			divisor = np.linalg.norm(thisData[j,6,:],thisData[j,7,:])
		thisData[j] = thisData/divisor
	trainData[i] = thisData
	print("Training Video %d root relatived", i)


for i in range(valLen):
        thisData = valData[i]
        thisData = thisData.reshape((-1,16,3))
        numFrames = thisData.shape[0]
        #divisor = None
        for j in range(numFrames):
		thisData[j,:,:] = thisData[j,:,:] - this[0,6,:]
		if j==0:
                        divisor = np.linalg.norm(thisData[j,6,:],thisData[j,7,:])
                thisData[j] = thisData/divisor
        trainData[i] = thisData
        print("Validation Video %d root relatived", i)


pickle.dump(trainData,open('../datasets/trainData.npy','wb'))
pickle.dump(valData,open('../datasets/valData.npy','wb'))

