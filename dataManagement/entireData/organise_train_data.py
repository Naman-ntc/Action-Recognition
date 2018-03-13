import numpy as np 
import pickle

###############################################################################################################3
stupid_videos  = [  822,   822,   822,  1837,  1837,  1837,  2154,  2154,  2154,
        2596,  2596,  2596,  3761,  3761,  3761,  4153,  4153,  4153,
        4429,  4429,  4429,  4833,  4833,  4833,  5047,  5047,  5047,
        5079,  5079,  5079,  5977,  5977,  5977,  6174,  6174,  6174,
        6290,  6290,  6290,  7097,  7097,  7097,  7141,  7141,  7141,
        8345,  8345,  8345,  9194,  9194,  9194,  9405,  9405,  9405,
        9726,  9726,  9726,  9793,  9793,  9793, 10290, 10290, 10290,
       10464, 10464, 10464, 10978, 10978, 10978, 12021, 12021, 12021,
       12757, 12757, 12757, 13576, 13576, 13576, 13606, 13606, 13606,
       13792, 13792, 13792, 14861, 14861, 14861, 15361, 15361, 15361,
       16144, 16144, 16144, 17055, 17055, 17055, 17129, 17129, 17129,
       17463, 17463, 17463, 18414, 18414, 18414, 18974, 18974, 18974,
       21016, 21016, 21016, 22692, 22692, 22692, 22881, 22881, 22881,
       22958, 22958, 22958, 23282, 23282, 23282, 23646, 23646, 23646,
       24636, 24636, 24636, 25129, 25129, 25129, 25926, 25926, 25926,
       26707, 26707, 26707, 27262, 27262, 27262, 28295, 28295, 28295,
       28634, 28634, 28634, 29075, 29075, 29075, 29356, 29356, 29356,
       30261, 30261, 30261, 30786, 30786, 30786, 32817, 32817, 32817,
       33066, 33066, 33066, 33370, 33370, 33370, 33545, 33545, 33545,
       34853, 34853, 34853, 35059, 35059, 35059, 35121, 35121, 35121,
       35280, 35280, 35280, 35663, 35663, 35663, 35923, 35923, 35923,
       36061, 36061, 36061, 36633, 36633, 36633, 37124, 37124, 37124,
       37557, 37557, 37557, 37593, 37593, 37593, 38506, 38506, 38506,
       38539, 38539, 38539, 38573, 38573, 38573, 38613, 38613, 38613,
       39138, 39138, 39138, 39418, 39418, 39418, 39530, 39530, 39530,
       39659, 39659, 39659, 39756, 39756, 39756]#xsub

stupid_videos = stupid_videos[0:len(stupid_videos):3]

train_data = np.load(open('train_data.npy','rb'))
train_labes = pickle.load(open('train_label.pkl','rb'))
print(train_data.shape)
non_stupid = np.setdiff1d(range(len(train_labes[1])),stupid_videos)
print(len(non_stupid))
train_data = train_data[non_stupid,:,:,:,:]
print(train_data.shape)
print(len(train_labes[1]))
train_labes = np.asarray(train_labes[1])
print(train_labes.shape)
train_labes = train_labes[non_stupid]
train_data = train_data[np.asarray(train_labes)<49,:,:,:,0]
print(train_data.shape)

#train_labes = np.asarray(train_labes[1])

#train_labes = train_labes[non_stupid]

train_labes = train_labes[train_labes<49]

np.save('Final-Data/train_data.npy',train_data)
np.save('Final-Data/train_labels.npy',train_labes)

# indices = [0,13,22,23,37]
# mask = np.zeros(train_labes.shape[0])
# for i in range(train_labes.shape[0]):
#   if train_labes[i] in indices:
#     mask[i] = 1

# train_data = train_data[mask == 1]
# train_labes = train_labes[mask == 1]

# for i in range(train_labes.shape[0]):
#   if train_labes[i] == 13:
#     train_labes[i] = 1
#   if train_labes[i] == 22:
#     train_labes[i] = 2
#   if train_labes[i] == 23:
#     train_labes[i] = 3
#   if train_labes[i] == 37:
#     train_labes[i] = 4

# np.save('toyData/trainData.npy', train_data)
# np.save('toyData/trainLabels.npy', train_labes)
