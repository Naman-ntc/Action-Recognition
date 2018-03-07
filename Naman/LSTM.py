import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from helperFunctions import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def to_categorical(y, num_classes):
	""" 1-hot encodes a tensor """
	return np.eye(num_classes, dtype='uint8')[y.astype(int)]


class LSTMClassifier(nn.Module):

	def __init__(self, hidden_dim=64,label_size=49,modified_input_dim=64):
		super(LSTMClassifier, self).__init__()
		self.hidden_dim = hidden_dim
		self.fully_connected = nn.Sequential(nn.Linear(75, 70),nn.ReLU(),nn.Linear(70, 64),nn.ReLU())
		self.lstm = nn.LSTM(modified_input_dim, hidden_dim)
		self.hidden2label = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()
		#self.num_frames = num_frames

	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(300,1, self.hidden_dim)),
				autograd.Variable(torch.zeros(300,1, self.hidden_dim)))

	def forward(self, joint_3d_vec):
		x = joint_3d_vec
		x = self.fully_connected(x.view(x.size()[0],x.size()[2]))
		x = x.view(x.size()[0],1,x.size()[1])
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.hidden2label(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs

trainingData = torch.from_numpy(getData())
labels = getLabels()
#indices = torch.from_numpy((labels.reshape(labels.shape[0])<5).dtype()).type(torch.LongTensor)
#indices = (torch.from_numpy(labels)<5).numpy()

number = int((labels<5).sum())
#indices = (labels<5)
# labels = labels[indices,:]
# trainingData = trainingData[indices,:,:,:]

neededData = torch.randn(number, 300, 25, 3)
neededLabels = np.zeros((number,1))
currentIndex = 0
for i in range(labels.shape[0]):
	if labels[i, 0] < 5:
		neededData[currentIndex,:,:,:] = trainingData[i,:,:,:]
		neededLabels[currentIndex,:] = labels[i,:]
		currentIndex+=1

#labels = torch.from_numpy(to_categorical((neededLabels),5)).view(number,-1)
labels = torch.from_numpy(neededLabels).view(number,-1).type(torch.cuda.LongTensor)
trainingData = neededData

def checkAcc(data,labels):
	l = labels.size()[0]
	labelsdash = autograd.Variable(labels.view(l))
	l = 1000
	out_labels = autograd.Variable(torch.zeros(l))
	for i in range(l):
		temp = model0(autograd.Variable(trainingData[i,:,:,:].view(300,1,75)))
		# print(temp)
		# print(temp.size(), type(temp))
		out_labels[i] = temp.max(1)[1]
	return(torch.mean((labelsdash[0:l].type(torch.cuda.LongTensor)==out_labels.type(torch.cuda.LongTensor)).type(torch.cuda.FloatTensor)))	

model0 = LSTMClassifier(label_size=5)


def TrainAcc():
	print(checkAcc(trainingData,labels))


#print(labels.size())
def train(model, num_epoch, num_iter, lr=1e-3,rec_interval=2, disp_interval=10):
	optimizer = optim.Adam(model.parameters(), lr)
	loss_values = []
	rec_step = 0
	for eph in range(num_epoch):
		print('epoch {} starting ...'.format(eph))
		avg_loss = 0
		n_samples = 0
		randpermed = torch.randperm(trainingData.size()[0])[:num_iter]
		for i in range(num_iter):
			model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
			model.zero_grad()
			
			j = randpermed[i]
			X,Y = trainingData[j,:,:,:].view(300,1,75),labels[j,:]
			n_samples += len(X)
			X = autograd.Variable(X)
			#print(X)
			Y = autograd.Variable(Y.view(1))
			y_hat = model(X)
			loss = F.cross_entropy(y_hat, Y)
			avg_loss += loss.data[0]
			if i % disp_interval == 0:
				print('epoch: %d iterations: %d loss :%g' % (eph, i, loss.data[0]))
			if rec_step%rec_interval==0:
				loss_values.append(loss.data[0])
			loss.backward()
			optimizer.step()
			rec_step += 1
		avg_loss /= n_samples
		#evaluating model accuracy
		#TrainAcc()
		print('epoch: {} <====train track===> avg_loss: {} \n'.format(eph, avg_loss))
	return loss_values



#l = train(model0, 10, 100, 2, 20)

def PlotLoss(l,name):
	plt.plot(l)			
	plt.show()
	plt.savefig(name)

def Scheduler():
	loss0 = train(model0,3,3300,6e-3)
	loss1 = train(model0,20,3300,1e-3)
	PlotLoss(loss1,'loss1.png')
	TrainAcc()
	loss2 = train(model0,20,3300,1e-3)
	TrainAcc()
	loss3 = train(model0,20,3300,1e-4)
	PlotLoss(loss1+loss2+loss3,'loss2.png')
	TrainAcc()
	loss4 = train(model0,20,3300,1e-4)
	TrainAcc()
	loss5 = train(model0,50,3300,1e-5)
	PlotLoss(loss1+loss2+loss3+loss4+loss5,'loss3.png')
	TrainAcc()