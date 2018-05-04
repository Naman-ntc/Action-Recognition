import numpy
import pickle
import torch

labelDict = pickle.load(open('labelDict.pkl','rb'))

trainDict = pickle.load(open('train_data.pkl'.'rb'))

total_samples = len(trainDict)

data = [None]*total_samples
labels = np.zeros(total_samples)

counter = 0

for k,v in trainDict:
	myk = int(k[5:])
	mylab = labelDict[myk]
	data[counter] = torch.from_numpy(v.reshape(-1))
	labels = labelDict[myk]
	counter += 1


pickle.dump(data,open('trainData.npy'))
np.save('trainLabels.npy', labels)