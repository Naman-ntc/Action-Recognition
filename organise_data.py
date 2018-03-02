import numpy as np 
import pickle

train_data = np.load(open('train_data.npy','rb'))
train_labes = pickle.load(open('small_label.pkl','rb'))

train_data = train_data[np.asarray(train_labes[1])<49,:,:,:,0]
train_labes = np.asarray(train_labes[1])
train_labes = train_labes[train_labes[1]<49]

train_data = train_data - (train_data[:,:,0,0])[:,:,None,None]
train_data = train_data / np.linalg.norm(train_data[:,:,0,1]-train_data[:,:,0,0],axis=1)[:,None,None,None]

val_data = np.load(open('small_val.npy','rb'))
val_labes = pickle.load(open('small_label.pkl','rb'))

val_data = val_data[np.asarray(val_labes[1])<49,:,:,:,0]
val_labes = np.asarray(val_labes[1])
val_labes = val_labes[val_labes[1]<49]

val_data = val_data - (val_data[:,:,0,0])[:,:,None,None]
val_data = val_data / np.linalg.norm(val_data[:,:,0,1]-val_data[:,:,0,0],axis=1)[:,None,None,None]


np.save('Final-Data/train_data.npy',train_data)
np.save('Final-Data/train_labels.npy',train_labes)
np.save('Final-Data/val_data.npy',val_data)
np.save('Final-Data/val_labes.npy',val_labes)