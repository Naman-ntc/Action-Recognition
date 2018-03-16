import Models
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1
import numpy as np
from keras.optimizers import SGD, RMSprop, Adam #change to adam
from keras.utils import to_categorical
from helperFunctions import *
from keras import regularizers as regularizer
from keras import metrics


class PlotLosses(keras.callbacks.Callback):	
	
	def __init__(self):
		super(PlotLosses, self).__init__()
		self.losses = []
		self.x = []

	def on_train_begin(self, logs={}):
		self.i = 0
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
		if (self.x[-1]%8 == 0):
			TrainAcc()	


trainingData = getData()
temp_shape = trainingData.shape 
trainingData = trainingData.reshape(temp_shape[0],300,-1)
labels = getLabels()
labels = to_categorical(labels,num_classes=49)

n_classes = 49
feat_dim = 75
max_len = 300

dropout = 0.5
reg = l1(6.e-5)
activation = "relu"

model = Models.TCN_resnet(
		n_classes,
		feat_dim,
		max_len,
		gap=1,
		dropout=dropout,
		kernel_regularizer=reg,
		activation=activation)


adam = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[metrics.categorical_crossentropy])

plot_losses = PlotLosses()

# c.model.fit(c.trainingData, c.labels, batch_size = c.batchSize, epochs = 100)
# c.checkAcc(c.model,c.trainingData,c.labels)

def train(batchSize=64,epochs=6):
	model.fit(trainingData, labels, batch_size = batchSize, epochs = epochs, callbacks=[plot_losses])
	TrainAcc()

def change_lr(new_lr):
	adam.lr = new_lr

def TrainAcc():
	print(checkAcc(model,trainingData,labels))

def ValAcc():
	trainingData = getValData()
	labels = getValLabels()
	temp_shape = trainingData.shape 
	trainingData = trainingData.reshape(temp_shape[0],300,-1)
	labels = to_categorical(labels,num_classes=49)
	print(checkAcc(model,trainingData,labels))	

def Schedule(l):
	for tup in l:
		change_lr(tup[0])
		train(epochs=tup[1])

  