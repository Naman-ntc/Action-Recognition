from cudaHelperFunctions import *


import pickle
data = getData()
myData = []
for i in range(128):
	myData.append(data[i].view(-1, 48)/200.0)

pickle.dump(myData, open("toyData.npy", 'wb'))

