import numpy as np 
import pickle

###############################################################################################################3
stupid_videos = [  491,   491,   491,   670,   670,   670,  1165,  1165,  1165,
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
       15807, 15807, 15807, 16174, 16174, 16174] #xsub

stupid_videos = stupid_videos[0:len(stupid_videos):3]

train_data = np.load(open('val_data.npy','rb'))
train_labes = pickle.load(open('val_label.pkl','rb'))
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

np.save('Final-Data/val_data.npy',train_data)
np.save('Final-Data/val_labels.npy',train_labes)

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
