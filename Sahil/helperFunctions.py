import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import keras

def getData():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_data.npy')
	model = np.swapaxes(np.swapaxes(model[:,:,:,:],1,3),1,2)
	return model

def getLabels():
	model = np.load('../datasets/NTU/xsub/Final-Data/train_labels.npy')
	labels = model
	labels = labels.reshape(-1,1)
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

class PlotLosses(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.i = 0
		self.x = []
		self.losses = []
		self.val_losses = []
		
		self.fig = plt.figure()
		
		self.logs = []

	def on_batch_end(self, epoch, logs={}):
		
		self.logs.append(logs)
		self.x.append(self.i)
		self.losses.append(logs.get('loss'))
		self.i += 1
		
		#clear_output(wait=True)
	
	def on_epoch_end(self, epoch, logs={}):
		plt.plot(self.x, self.losses, label="loss")
		plt.legend()
		plt.show()
		plt.savefig('current_loss.png')
		plt.close()
		
