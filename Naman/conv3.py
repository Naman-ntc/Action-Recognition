from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam #change to adam
from keras.utils import to_categorical
from helperFunctions import *
from keras import regularizer


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
		TrainAcc()	

learningRate = 1e-3
batchSize = 64
regParam = 0.01

trainingData = getData()
labels = getLabels()
labels = to_categorical(labels,num_classes=49)

model = Sequential()
model.add(BatchNormalization(input_shape=(300,25,3)))
model.add(Conv2D(32, (3,3), strides=(2,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), strides=(2,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), strides=(2,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), strides=(2,1), padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(regParam)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(2*regParam)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(2*regParam)))
model.add(BatchNormalization())
model.add(Dense(49, activation='softmax', kernel_initializer='he_normal', kernel_regularizer = regularizer.l2(2*regParam)))

print(model.summary())

#print(checkAcc(model,trainingData,labels))

#rmsprop = RMSprop(lr=learningRate, rho=0.9, epsilon=None, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam)

plot_losses = PlotLosses()

# c.model.fit(c.trainingData, c.labels, batch_size = c.batchSize, epochs = 100)
# c.checkAcc(c.model,c.trainingData,c.labels)

def train(batch_size=64,epochs=6):
	model.fit(trainingData, labels, batch_size = batchSize, epochs = epochs, callbacks=[plot_losses])
	TrainAcc()

def change_lr(new_lr):
	adam.lr = new_lr

def TrainAcc():
	print(checkAcc(model,trainingData,labels))

def ValAcc():
	trainingData = getValData()
	labels = getValLabels()
	labels = to_categorical(labels,num_classes=49)
	print(checkAcc(model,trainingData,labels))	

def Schedule(l):
	for tup in l:
		change_lr(tup[0])
		train(epochs=tup[1])

