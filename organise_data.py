import numpy as np 
import pickle


###############################################################################################################3
stupid_videos = [  822,   822,   822,  1837,  1837,  1837,  2154,  2154,  2154,
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
       39659, 39659, 39659, 39756, 39756, 39756] #xsub

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

train_data = train_data - (train_data[:,:,0,0])[:,:,None,None]
print(train_data.shape)

train_data = train_data / np.linalg.norm(train_data[:,:,0,1]-train_data[:,:,0,0],axis=1)[:,None,None,None]
print(train_data.shape)


#####################################################################################################################

stupid_videos = [491,   491,   491,   670,   670,   670,  1165,  1165,  1165,
        1297,  1297,  1297,  2371,  2371,  2371,  2742,  2742,  2742,
        2920,  2920,  2920,  3098,  3098,  3098,  3182,  3182,  3182,
        3382,  3382,  3382,  3624,  3624,  3624,  3963,  3963,  3963,
        4030,  4030,  4030,  5149,  5149,  5149,  5804,  5804,  5804,
        6747,  6747,  6747,  6807,  6807,  6807,  7073,  7073,  7073,
        7323,  7323,  7323,  7355,  7355,  7355,  7697,  7697,  7697,
        7966,  7966,  7966,  8075,  8075,  8075,  8167,  8167,  8167,
        8570,  8570,  8570,  9306,  9306,  9306,  9626,  9626,  9626,
       10156, 10156, 10156, 10569, 10569, 10569, 11095, 11095, 11095,
       11241, 11241, 11241, 11418, 11418, 11418, 11620, 11620, 11620,
       11688, 11688, 11688, 12129, 12129, 12129, 12644, 12644, 12644,
       13503, 13503, 13503, 13735, 13735, 13735, 13856, 13856, 13856,
       14766, 14766, 14766, 15100, 15100, 15100, 15251, 15251, 15251,
       15807, 15807, 15807, 16174, 16174, 16174]

#non_stupid = np.setdiff1d(range(len(val_labes[1])),stupid_videos)

val_data = np.load(open('val_data.npy','rb'))
val_labes = pickle.load(open('val_label.pkl','rb'))

non_stupid = np.setdiff1d(range(len(val_labes[1])),stupid_videos)

val_labes = np.asarray(val_labes[1])
val_labes = val_labes[non_stupid]

val_data = val_data[np.asarray(val_labes)<49,:,:,:,0]

val_labes = val_labes[val_labes<49]

val_data = val_data - (val_data[:,:,0,0])[:,:,None,None]
val_data = val_data / np.linalg.norm(val_data[:,:,0,1]-val_data[:,:,0,0],axis=1)[:,None,None,None]

"""
np.save('Final-Data/train_data.npy',train_data)
np.save('Final-Data/train_labels.npy',train_labes)
np.save('Final-Data/val_data.npy',val_data)
np.save('Final-Data/val_labes.npy',val_labes)
"""
