import numpy as np 
import pickle


#####################################################################################################################

stupid_videos = [  111,   111,   111,   747,   747,   747,   981,   981,   981,
        1145,  1145,  1145,  1252,  1252,  1252,  1281,  1281,  1281,
        1282,  1282,  1282,  1485,  1485,  1485,  1504,  1504,  1504,
        1840,  1840,  1840,  1865,  1865,  1865,  1916,  1916,  1916,
        2071,  2071,  2071,  2220,  2220,  2220,  3108,  3108,  3108,
        4133,  4133,  4133,  4507,  4507,  4507,  4882,  4882,  4882,
        5081,  5081,  5081,  5293,  5293,  5293,  5315,  5315,  5315,
        5643,  5643,  5643,  5816,  5816,  5816,  6082,  6082,  6082,
        6648,  6648,  6648,  6695,  6695,  6695,  6773,  6773,  6773,
        6873,  6873,  6873,  7137,  7137,  7137,  7616,  7616,  7616,
        7680,  7680,  7680,  9472,  9472,  9472,  9533,  9533,  9533,
       10120, 10120, 10120, 10588, 10588, 10588, 11693, 11693, 11693,
       12150, 12150, 12150, 12218, 12218, 12218, 13542, 13542, 13542,
       13860, 13860, 13860, 14701, 14701, 14701, 14935, 14935, 14935,
       16026, 16026, 16026, 16298, 16298, 16298]
#non_stupid = np.setdiff1d(range(len(val_labes[1])),stupid_videos)

val_data = np.load(open('val_data.npy','rb'))
val_labes = pickle.load(open('val_label.pkl','rb'))
print(val_data.shape)
print(len(val_labes[1]))
non_stupid = np.setdiff1d(range(len(val_labes[1])),stupid_videos)

val_labes = np.asarray(val_labes[1])
val_labes = val_labes[non_stupid]
print(len(val_labes))
val_data = val_data[non_stupid,:,:,:,:]

val_data = val_data[np.asarray(val_labes)<49,:,:,:,0]
print(val_data.shape)

val_labes = val_labes[val_labes<49]

val_data = val_data - (val_data[:,:,0,0])[:,:,None,None]
val_data = val_data / np.linalg.norm(val_data[:,:,0,1]-val_data[:,:,0,0],axis=1)[:,None,None,None]


np.save('Final-Data/val_data.npy',val_data)
np.save('Final-Data/val_labes.npy',val_labes)

