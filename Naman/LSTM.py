import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helperFunctions import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


class LSTMClassifier(nn.Module):

	def __init__(self, hidden_dim=64,label_size=49,modified_input_dim=64):
		super(LSTMClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.fully_connected = nn.Sequential(nn.Linear(75, 70),nn.ReLU(),nn.Linear(70, 64),nn.ReLU(),nn.Linear(64, 64),nn.ReLU())
		self.lstm = nn.LSTM(modified_input_dim, hidden_dim)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()
		self.num_frames = num_frames

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(1,1, self.hidden_dim)),
				autograd.Variable(torch.zeros(1,1, self.hidden_dim)))

	def forward(self, joint_3d_vec):
		x = joint_3d_vec
		x = self.fully_connected(x.view(x.size()[0],x.size()[2]))
		x = x.view(x.size[0],1,x.size[1])
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.hidden2label(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs

trainingData = getData().from_numpy()
labels = getLabels()
labels = to_categorical(labels,num_classes=49).from_numpy()

indices = (labels<5)
labels = labels[indices,:]
trainingData = trainingData[indices,:,:,:]

def train(model, num_epoch, num_iter, rec_interval, disp_interval):
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    loss_values = []
    rec_step = 0
    for eph in range(num_epoch):
        print('epoch {} starting ...'.format(eph))
        avg_loss = 0
        n_samples = 0
        torch.randperm(trainingData.size()[0])[:num_iter]
        for i in range(num_iter):
        	j = trainingData[i]
            X,Y = trainingData[j,:,:,:].view(300,1,75),labels[j,:]
            n_samples += len(X)
            X = autograd.Variable(torch.from_numpy(X).float())
            #print(X)
            Y = autograd.Variable(torch.LongTensor(np.array([Y])))
            y_hat = model(X)
            model.zero_grad()
            loss = F.cross_entropy(y_hat, Y)
            avg_loss += loss.data[0]
            if i % disp_interval == 0:
                print('epoch: %d iterations: %d loss :%g\n' % (eph, i, loss.data[0]))
            if rec_step%rec_interval==0:
                loss_values.append(loss.data[0])
            model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
            loss.backward()
            optimizer.step()
            rec_step += 1
        avg_loss /= n_samples
        #evaluating model accuracy
        acc = evaluate_accuracy(model, test_split)
        print('epoch: {} <====train track===> avg_loss: {}, accuracy: {}% \n'.format(eph, avg_loss, acc))
    return loss_values


model0 = LSTMClassifier(label_size=5)
#l = train(model0, 10, 100, 2, 20)

plt.plot(l)			
plt.show()
plt.savefig('loss.png')

def checkAcc(data,labels):
	l = labels.size()[0]
	out_labels = torch.zeros(l,1)
	for i in range(l):
		out_labels[i,:] = model.forward(trainingData[i,:,:,:].view(300,1,75)).max()[1]
	return(torch.mean(labels.max(dim=1)[1]==out_labels))	

def TrainAcc():
	print(checkAcc(trainingData,labels))
