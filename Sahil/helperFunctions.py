import matplotlib
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import autograd

def getData():
	model = pickle.load(open('../datasets/toyData/lstmProcessedTrainData.npy', 'rb'))
	return model

def getLabels():
	labels = np.load('../datasets/toyData/trainLabels.npy')
	return torch.from_numpy(labels).type(torch.LongTensor)


def getValData():
	model = pickle.load(open('../datasets/toyData/lstmProcessedValData.npy', 'rb'))
	return model

def getValLabels():
	labels = np.load('../datasets/toyData/valLabels.npy')
	return torch.from_numpy(labels).type(torch.LongTensor)

# def checkAcc(model, data, labels):
# 	pred = model.predict(data)
# 	return np.mean(np.argmax(pred,axis=1) == np.argmax(labels,axis=1))



def checkAcc(model0,data,labels, length = 1000):
	if length == -1:
		l = labels.size()[0]
	else:
		l = length
		labels = labels[:l]
	labelsdash = autograd.Variable(labels.view(l))
	out_labels = autograd.Variable(torch.zeros(l))
	for i in range(l):
		temp = model0(autograd.Variable(data[i].view(data[i].size()[0],1,75)))
		# print(temp)
		# print(temp.size(), type(temp))
		out_labels[i] = temp.max(1)[1]
	return(torch.mean((labelsdash[0:l].type(torch.LongTensor)==out_labels.type(torch.LongTensor)).type(torch.FloatTensor)))	



def PlotLoss(l,name = 'currentLoss.png'):
	plt.plot(l)			
	plt.show()
	plt.savefig(name)
