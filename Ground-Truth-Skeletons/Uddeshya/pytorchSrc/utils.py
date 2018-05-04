import random, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#making test and train split
def train_test_split(dframe_list):
    train_split = np.empty(0, dtype=object)
    test_split = np.empty(0, dtype=object)
    for dframe in dframe_list:
        label = dframe.iloc[0,75]
        num_samples = len(dframe.iloc[:,:])
        video_ids = np.unique(dframe.iloc[:,-1].values)
        train_video_ids = video_ids[:-15]
        test_video_ids = video_ids[-15:]
        train_split1 = np.empty(len(train_video_ids), dtype=object)
        test_split1 = np.empty(len(test_video_ids), dtype=object)
        for idx,i in enumerate(train_video_ids):
            train_split1[idx] = dframe.loc[dframe['video_id'] == i].values[:,0:75]
            for fidx, f in enumerate(train_split1[idx]):
                f = np.reshape(f, (25,3))
                f = f-f[0,:]
                f = np.reshape(f, (1,75))
                train_split1[idx][fidx] = f
            train_split1[idx] = (train_split1[idx], label)

        for idx,i in enumerate(test_video_ids):
            test_split1[idx] = dframe.loc[dframe['video_id'] == i].values[:,0:75]
            for fidx, f in enumerate(test_split1[idx]):
                f = np.reshape(f, (25,3))
                f = f-f[0,:]
                f = np.reshape(f, (1,75))
                test_split1[idx][fidx] = f
            test_split1[idx] = (test_split1[idx], label)
        train_split = np.concatenate((train_split, train_split1))
        test_split = np.concatenate((test_split, test_split1))
    return train_split, test_split

def Data_gen( train_split, SEQ_LEN):
    while(True):
        X = train_split
        databatch = random.sample(list(X), 1)[0]
        databatch, label = databatch[0], databatch[1]
        if SEQ_LEN is not None:
            if len(databatch) > SEQ_LEN:
                databatch = databatch[0:SEQ_LEN]
            elif len(databatch) < SEQ_LEN:
                databatch = np.concatenate((databatch, np.zeros((SEQ_LEN - len(databatch), 75))))
            else:
                pass
            yield databatch,label
        else:
            yield databatch,label