import matplotlib
import pickle
import numpy as np 
import matplotlib.pyplot as plt

def getData():
	model = pickle.load(open('../datasets/toyData/trainData.npy', 'rb'))
	return model

def getLabels():
	model = np.load('../datasets/toyData/trainLabels.npy')\
	return labels


def getValData():
	model = pickle.load(open('../datasets/NTU/toyData/valData.npy', 'rb'))
	return model

def getValLabels():
	model = np.load('../datasets/NTU/toyData/valLabels.npy')
	return labels

# def checkAcc(model, data, labels):
# 	pred = model.predict(data)
# 	return np.mean(np.argmax(pred,axis=1) == np.argmax(labels,axis=1))



def checkAcc(model0,data,labels, length = -1):
	if length==-1:
		l = labels.size()[0]
	else:
		l = length
	labelsdash = labels.view(l)
	out_labels = torch.zeros(l)
	for i in range(l):
		temp = model0(data[i,:,:,:].view(300,1,75))
		# print(temp)
		# print(temp.size(), type(temp))
		out_labels[i] = temp.max(1)[1]
	return(torch.mean((labelsdash[0:l].type(torch.cuda.LongTensor)==out_labels.type(torch.cuda.LongTensor)).type(torch.cuda.FloatTensor)))	

