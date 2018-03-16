import matplotlib
import pickle
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import autograd

def getData():
	model = pickle.load(open('../datasets/processedToyData/lstmProcessedTrainData.npy', 'rb'))
	for i in range(len(model)):
		model[i] = model[i].type(torch.cuda.FloatTensor)
	return model

def getLabels():
	labels = np.load('../datasets/processedToyData/trainLabels.npy')
	return torch.from_numpy(labels).type(torch.cuda.LongTensor)


def getValData():
	model = pickle.load(open('../datasets/processedToyData/lstmProcessedValData.npy', 'rb'))
	for i in range(len(model)):
		model[i] = model[i].type(torch.cuda.FloatTensor)
	return model

def getValLabels():
	labels = np.load('../datasets/processedToyData/valLabels.npy')
	return torch.from_numpy(labels).type(torch.cuda.LongTensor)

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
		temp = model0(data[i])
		# print(temp)
		# print(temp.size(), type(temp))
		if i%100 == 0:
			print("checking", i, "of", length)
		out_labels[i] = temp.max(1)[1]
	return(torch.mean((labelsdash[0:l].type(torch.cuda.LongTensor)==out_labels.type(torch.cuda.LongTensor)).type(torch.cuda.FloatTensor)))	



def PlotLoss(l,name = 'currentLoss.png'):
	plt.clf()
	plt.cla()
	plt.close()
	plt.plot(l)			
	plt.show()
	plt.savefig(name)
