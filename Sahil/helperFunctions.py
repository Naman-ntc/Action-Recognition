import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import keras

def getData():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_data.npy')
	model = np.swapaxes(np.swapaxes(model[:,:,:,:],1,3),1,2)[:300, :, :, :]
	return model

def getLabels():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_labels.npy')
	labels = model
	labels = labels.reshape(-1,1)[:300, :]
	return labels

def getValData():
	model = np.load('../datasets/NTU/xsub/Final-Data/val_data.npy')
	model = np.swapaxes(np.swapaxes(model[:,:,:,:],1,3),1,2)
	return model

def getValLabels():
	model = np.load('../datasets/NTU/xsub/Final-Data/val_labels.npy')
	labels = model
	labels = labels.reshape(-1,1)
	return labels

def checkAcc(model, data, labels):
	pred = model.predict(data)
	return np.mean(np.argmax(pred,axis=1) == np.argmax(labels,axis=1))

