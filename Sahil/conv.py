from keras import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D
from keras.layers import GLobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam #change to adam

from helperFunctions import *

learningRate = 1e-3
batchSize = 32

trainingData = getData()
labels = getLabels()

model = Sequential()
model.add(Conv2D(32, (3,3), padding=(1,1), activation='relu', input_shape=(1, None, None)))
model.add(Conv2D(32, (3,3), padding=(1,1), activation='relu'))
model.add(Conv2D(64, (3,3), padding=(1,1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(64, (3,3), padding=(1,1), activation='relu'))
model.add(Conv2D(64, (3,3), padding=(1,1), activation='relu'))
model.add(Conv2D(128, (3,3), padding=(1,1), activation='relu'))
model.add(Conv2D(128, (3,3), padding=(1,1), activation='relu'))
model.add(GLobalAveragePooling2D(data_format-'channels_first'))
model.add(Dense(128), activation='relu')
model.add(Dense(64), activation='relu')
model.add(Dense(48), activation='softmax')

rmsprop = keras.optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
model.fit(trainingData, labels, batch_size = batchSize, epochs = 10)
