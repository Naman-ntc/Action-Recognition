import pandas as pd
from random import random

flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:] * random()  # some noise

import numpy as np

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = 2
hidden_neurons = 50

model = Sequential()

# n_prev = 100, 2 values per x axis
model.add(LSTM(hidden_neurons, input_shape=(100, 2)))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))

model.compile(loss="mean_squared_error",
    optimizer="rmsprop",
    metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = train_test_split(data)

model.fit(X_train, y_train, batch_size=700, nb_epoch=50, validation_data=(X_test, y_test), verbose=1)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict(X_test, batch_size=700)

# and maybe plot it
pd.DataFrame(predicted).to_csv("predicted.csv")
pd.DataFrame(y_test).to_csv("test_data.csv")