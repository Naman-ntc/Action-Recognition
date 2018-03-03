from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, Flatten
from keras.optimizers import SGD, RMSprop, Adam #change to adam
from keras.utils import to_categorical
from helperFunctions import *

learningRate = 1e-3
batchSize = 64

trainingData = getData()
labels = getLabels()
labels = to_categorical(labels,num_classes=49)

model = Sequential()
model.add(BatchNormalization(input_shape=(300,25,3)))
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
# model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(49, activation='softmax'))

print(model.summary())

#print(checkAcc(model,trainingData,labels))

rmsprop = RMSprop(lr=learningRate, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

plot_losses = PlotLosses()

# c.model.fit(c.trainingData, c.labels, batch_size = c.batchSize, epochs = 100)
# c.checkAcc(c.model,c.trainingData,c.labels)

def train(batch_size=64,epochs=6):
	model.fit(trainingData, labels, batch_size = batchSize, epochs = epochs, callbacks=[plot_losses])
	TrainAcc()

def change_lr(new_lr):
	rmsprop.lr = new_lr

def TrainAcc():
	print(checkAcc(model,trainingData,labels))

def ValAcc():
	trainingData = getValData()
	labels = getValLabels()
	labels = to_categorical(labels,num_classes=49)
	print(checkAcc(model,trainingData,labels))	