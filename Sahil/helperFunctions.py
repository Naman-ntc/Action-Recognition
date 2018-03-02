import pickle
import numpy as np 

def getData():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_data.npy')
	model = model[:300,:,:,:]
	return model

def getLabels():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_labels.npy')
	labels = labels[:300]
	return labels

def getValData():
	model = np.load('../datasets/NTU/xsub/Final-Data/val_data.npy')
	model = model[:300,:,:,:]
	return model

def getValLabels():
	model = np.load('../datasets/NTU/xsub/Final-Data/val_labels.npy')
	labels = labels[:300]
	return labels

def checkAcc(model, data, labels):
	pred = model(data)
	return np.mean(pred == labels)