
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random, numpy as np
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(1)


# ## Loading the datasets, i.e loading frames for few actions

# In[2]:


#loading and prepping data
#initially only one action
dframe = pd.read_csv('./csv_data/action_1.csv')
dframe2 = pd.read_csv('./csv_data/action_2.csv')
dframe3 = pd.read_csv('./csv_data/action_3.csv')
dframe4 = pd.read_csv('./csv_data/action_4.csv')
dframe5 = pd.read_csv('./csv_data/action_5.csv')
dframe6 = pd.read_csv('./csv_data/action_6.csv')
dframe7 = pd.read_csv('./csv_data/action_7.csv')

dframe8 = pd.read_csv('./csv_data/action_8.csv')
dframe9 = pd.read_csv('./csv_data/action_9.csv')
dframe10 = pd.read_csv('./csv_data/action_10.csv')
dframe11 = pd.read_csv('./csv_data/action_11.csv')
dframe12 = pd.read_csv('./csv_data/action_12.csv')
dframe13 = pd.read_csv('./csv_data/action_13.csv')
dframe14 = pd.read_csv('./csv_data/action_14.csv')

dframe15 = pd.read_csv('./csv_data/action_15.csv')
dframe16 = pd.read_csv('./csv_data/action_16.csv')
dframe17 = pd.read_csv('./csv_data/action_17.csv')
dframe18 = pd.read_csv('./csv_data/action_18.csv')
dframe19 = pd.read_csv('./csv_data/action_19.csv')
dframe20 = pd.read_csv('./csv_data/action_20.csv')
dframe21 = pd.read_csv('./csv_data/action_21.csv')

#to look at data
dframe.iloc[0:5, :]


# ## Some utility functions to split the datasets and loading the datasets in batch

# In[3]:


#making test and train split
#the recentering has been done so that the pelvic joint is always at the origin
#labels are to be zero indexed
def train_test_split(dframe_list):
    train_split = np.empty(0, dtype=object)
    test_split = np.empty(0, dtype=object)
    for dframe in dframe_list:
        label = dframe.iloc[0,75]-1
#         print(label)
        num_samples = len(dframe.iloc[:,:])
        video_ids = np.unique(dframe.iloc[:,-1].values)
        train_video_ids = video_ids[:-35]
        test_video_ids = video_ids[-35:]
        train_split1 = np.empty(len(train_video_ids), dtype=object)
        test_split1 = np.empty(len(test_video_ids), dtype=object)
        for idx,i in enumerate(train_video_ids):
            train_split1[idx] = dframe.loc[dframe['video_id'] == i].values[:,0:75]
            for fidx, f in enumerate(train_split1[idx]):
                f = np.reshape(f, (25,3))
                f = f-f[0,:]
                f = np.reshape(f, (1,75))
                train_split1[idx][fidx] = f
#             mean_vec = np.mean(train_split1[idx], axis=0)
#             std_vec = np.std(train_split1[idx], axis=0)
            train_split1[idx] = (train_split1[idx], label)

        for idx,i in enumerate(test_video_ids):
            test_split1[idx] = dframe.loc[dframe['video_id'] == i].values[:,0:75]
            for fidx, f in enumerate(test_split1[idx]):
                f = np.reshape(f, (25,3))
                f = f-f[0,:]
                f = np.reshape(f, (1,75))
                test_split1[idx][fidx] = f
#             mean_vec = np.mean(test_split1[idx], axis=0)
#             std_vec = np.std(test_split1[idx], axis=0)
            test_split1[idx] = (test_split1[idx], label)
        train_split = np.concatenate((train_split, train_split1))
        test_split = np.concatenate((test_split, test_split1))
    return train_split, test_split

train_split, test_split = train_test_split([dframe, dframe2, dframe3, dframe4, dframe5, dframe6, dframe7, dframe8, dframe9,
                                           dframe10, dframe11, dframe12, dframe13, dframe14, dframe15, dframe16, dframe17,
                                           dframe19, dframe20, dframe21])

# #looking at split
train_split[0:3]


# In[4]:


SEQ_LEN = None
def Data_gen( train_split, SEQ_LEN):
    while(True):
        X = train_split
        databatch = random.sample(list(X), 1)[0]
#         print(databatch)
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

ACTd = Data_gen(train_split, SEQ_LEN)

#to look at batch created by Actd
next(ACTd)


# ## LSTM Classifier model defination and intialisation

# In[5]:


#action LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, joints_dim, hidden_dim, label_size, batch_size, num_layers, kernel_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        joints_dim2d = joints_dim - 25
        
        self.lstm3 = nn.LSTM(joints_dim, hidden_dim, num_layers=self.num_layers)
        self.conv1_3 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
        
        self.lstm2_1 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
        self.conv1_2_1 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
        self.lstm2_2 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
        self.conv1_2_2 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
        self.lstm2_3 = nn.LSTM(joints_dim2d, hidden_dim, num_layers=self.num_layers)
        self.conv1_2_3 = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)
        
        self.dense3 = nn.Linear(hidden_dim, joints_dim)
        self.dense2_1 = nn.Linear(hidden_dim, joints_dim2d)
        self.dense2_2 = nn.Linear(hidden_dim, joints_dim2d)
        self.dense2_3 = nn.Linear(hidden_dim, joints_dim2d)
        
#         self.conv1_1 = nn.Conv1d(4, 2, kernel_size, stride=1, padding=1) #for kernel size=3
#         self.conv1_2 = nn.Conv1d(2, 1, kernel_size, stride=1, padding=1) #for kernel size=3
        
        self.hidden3 = self.init_hidden3()
        self.hidden2_1 = self.init_hidden2_1()
        self.hidden2_2 = self.init_hidden2_2()
        self.hidden2_3 = self.init_hidden2_3()
        
        self.hidden2label = nn.Linear(hidden_dim, label_size)
    
    def init_hidden3(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
    def init_hidden2_1(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
    def init_hidden2_2(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
    def init_hidden2_3(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda()))
    
    
    def forward(self, joints3d_vec):
        x3 = joints3d_vec
#         print('x3 : ', x3.size())
        x2 = x3.view(-1, 25, 3)
        x2_1 = x2[:,:,1:3].contiguous().view(-1, 1, 50)
        x2_2 = x2[:,:,0:2].contiguous().view(-1, 1, 50)
        x2_3 = x2[:,:,[0,2]].contiguous().view(-1, 1, 50)
#         print('x2_3 : ',x2_3.size())
        lstm_out3, self.hidden3 = self.lstm3(x3, self.hidden3)
        lstm_out2_1, self.hidden2_1 = self.lstm2_1(x2_1, self.hidden2_1)
        lstm_out2_2, self.hidden2_2 = self.lstm2_2(x2_2, self.hidden2_2)
        lstm_out2_3, self.hidden2_3 = self.lstm2_3(x2_3, self.hidden2_3)
#         print('lstm_out[-1] : ', lstm_out[-1].size())
        t3 = lstm_out3[-1].view(self.batch_size,1,-1)
#         print('t3 : ', t3.size())
        t2_1 = lstm_out2_1[-1].view(self.batch_size,1,-1)
        t2_2 = lstm_out2_2[-1].view(self.batch_size,1,-1)
        t2_3 = lstm_out2_3[-1].view(self.batch_size,1,-1)
#         print('t2_3 : ', t2_3.size())
        
        y3 = self.conv1_3(t3)
#         print('y3 : ', y3.size())
        y2_1 = self.conv1_2_1(t2_1)
#         print('y2_1 : ', y2_1.size())
        y2_2 = self.conv1_2_2(t2_2)
#         print('y2_2 : ', y2_2.size())
        y2_3 = self.conv1_2_3(t2_3)
#         print('y2_3 : ', y2_3.size())
        
        yf = y3+y2_1+y2_2+y2_3
        
        yf = yf.contiguous().view(-1, self.hidden_dim)
#         print('y3 : ', y3.size())
        y  = self.hidden2label(yf)
    
        ## for outputs of other timesteps
        t_prime3 = lstm_out3
        t_prime2_1 = lstm_out2_1
        t_prime2_2 = lstm_out2_2
        t_prime2_3 = lstm_out2_3
#         print('t_prime3 : ', t_prime3.size())
#         print('t_prime3[0] : ', t_prime3[0].size())
#         print('coming for loop .. ')
        y3_others = autograd.Variable(torch.zeros((t_prime3.size()[0], t_prime3.size()[1], 75)).cuda())
        y2_1_others = autograd.Variable(torch.zeros((t_prime2_1.size()[0], t_prime2_1.size()[1], 50)).cuda())
        y2_2_others = autograd.Variable(torch.zeros((t_prime2_2.size()[0], t_prime2_2.size()[1], 50)).cuda())
        y2_3_others = autograd.Variable(torch.zeros((t_prime2_3.size()[0], t_prime2_3.size()[1], 50)).cuda())
#         print('t_future3 : ', t_future3.size())
        for idx,tp in enumerate(t_prime3):
            y3_others[idx, :, :] = self.dense3(tp)
        for idx,tp in enumerate(t_prime2_1):
            y2_1_others[idx, :, :] = self.dense2_1(tp)
        for idx,tp in enumerate(t_prime2_2):
            y2_2_others[idx, :, :] = self.dense2_2(tp)
        for idx,tp in enumerate(t_prime3):
            y2_3_others[idx, :, :] = self.dense2_3(tp)
#         print('y_2_3_others : ', y2_3_others.size())
#         print('y3_others : ', y3_others.size())
        log_probs = F.log_softmax(y, dim=1)
        return log_probs, y3_others, y2_1_others, y2_2_others, y2_3_others #others as in the output from the cells behind the last cell
#instanstiating a model
model0 = LSTMClassifier(75, 512, 7, 1, 2, 3)
#to do stuff in CUDA
model0 = model0.cuda()


# ## Training the model

# In[6]:


def evaluate_accuracy(model, test_split):
    pred_labels = np.empty(len(test_split))
    orig_labels = np.array([t[1] for t in test_split])
    for i in range(len(test_split)):
        d_in = autograd.Variable(torch.from_numpy(test_split[i][0]).float().cuda())
        d_in = d_in.view(d_in.size()[0], 1, -1)
        y_pred = model(d_in)
        pred_labels[i] = y_pred.data.cpu().max(1)[1].numpy()[0];
    n_samples = len(pred_labels)
    res=(orig_labels==pred_labels)
    correct_count = (res==True).sum()
    return (correct_count*100/n_samples)


# ## observations
# * better to use the log_softmax instead of softmax
# * decrease lr succicesively to get better results

# In[7]:


#training function
def train(model, num_epoch, num_iter, rec_interval, disp_interval):
    optimizer = optim.Adam(model.parameters(), lr = 8e-6)
    loss_values = []
    avg_loss_values = []
    rec_step = 0
    print('Starting the training ...')
    torch.cuda.synchronize()
    for eph in range(num_epoch):
        print('epoch {} starting ...'.format(eph))
        avg_loss = 0
        n_samples = 0
        torch.cuda.synchronize()
        for i in range(num_iter):
            model.hidden3 = (model.hidden3[0].detach(), model.hidden3[1].detach())
            model.hidden2_1 = (model.hidden2_1[0].detach(), model.hidden2_1[1].detach())
            model.hidden2_2 = (model.hidden2_2[0].detach(), model.hidden2_2[1].detach())
            model.hidden2_3 = (model.hidden2_3[0].detach(), model.hidden2_3[1].detach())
            model.zero_grad()
            X,Y = next(ACTd)
            n_samples += len(X)
            X = autograd.Variable(torch.from_numpy(X).float().cuda())
            X = X.view(len(X), 1, -1)
            Y = autograd.Variable(torch.LongTensor(np.array([Y])).cuda())
            
            X2 = X.view(-1, 25, 3)
            X2_1 = X2[:,:,1:3].contiguous().view(-1, 1, 50)
            X2_2 = X2[:,:,0:2].contiguous().view(-1, 1, 50)
            X2_3 = X2[:,:,[0,2]].contiguous().view(-1, 1, 50)
            
            y_hat, cord3, cord2_1, cord2_2, cord2_3 = model(X)
            print('cord3 : {}'.format(cord3.size()))
            print('X : {}'.format(X.size()))
            print('cord3 : {}'.format(cord3.size()))
            print('Y : ', Y)
            
            loss_classification = F.cross_entropy(y_hat, Y)
            loss_regr3d = F.mse_loss(cord3, X)
#             print('loss_regr3d : {}'.format(loss_regr3d))
            
            loss_regr2d_1 = F.mse_loss(cord2_1, X2_1)
            loss_regr2d_2 = F.mse_loss(cord2_2, X2_2)
            loss_regr2d_3 = F.mse_loss(cord2_3, X2_3)
            
            loss = loss_classification + loss_regr3d + loss_regr2d_1 + loss_regr2d_2 + loss_regr2d_3
#             print(loss)
            avg_loss += loss.data[0]
            
            if i % disp_interval == 0:
                print('epoch: %d iterations: %d loss :%g' % (eph, i, loss.data[0]))
            if rec_step%rec_interval==0:
                loss_values.append(loss.data[0])
            
            loss.backward()     
            optimizer.step()
            rec_step += 1
            
        avg_loss /= n_samples
        avg_loss_values.append(avg_loss)
        #evaluating model accuracy
        acc = evaluate_accuracy(model, test_split)
        print('epoch: {} <====train track===> avg_loss: {}, accuracy: {}% \n'.format(eph, avg_loss, acc))
    return loss_values, avg_loss_values


loss_vals, avg_loss_vals = train(model0, 100, 1000, 2, 100)
plt.figure()
plt.plot(loss_vals)
plt.figure()
plt.plot(avg_loss_vals)
plt.xlabel('epoch')
plt.ylabel('avg loss')


# In[9]:


a = autograd.Variable(torch.rand(3,5).cuda())
a


# In[30]:


# saving the model
def save_model(model_name, path, model):
    p = path+'/'+model_name
    print('saving at {}'.format(p))
    torch.save(model.state_dict(), p)
    print('saved at {}'.format(p))


# In[31]:


save_model('LSTMClassifierX1_c7.pth', './checkpoints', model0)


# In[33]:


mtest = LSTMClassifier(75, 512, 7, 1, 2, 3).cuda()
mtest.load_state_dict(torch.load('./checkpoints/LSTMClassifierX1_c7.pth'))


# In[39]:


mtest.lstm.weight_ih_l0

