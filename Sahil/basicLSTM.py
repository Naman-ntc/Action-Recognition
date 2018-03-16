import torch
from helperFunctions import *
import torch.nn as nn
from torch import autograd
from torch import optim
import torch.nn.functional as F

def TrainAcc():
	print(checkAcc(model,data,labels, length = 1000)[0])

def ValAcc():
	print(checkAcc(model, valData, valLabels)[0])

class LSTMClassifier(nn.Module):

	def __init__(self, hidden_dim=128, label_size=49, input_dim=75, num_layers = 1):
		super(LSTMClassifier, self).__init__()
		self.hiddenDim = hidden_dim
		self.layers = num_layers
		self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers = num_layers)
		self.fullyConnected = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()
		
	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.layers, 1, self.hiddenDim)),
				autograd.Variable(torch.zeros(self.layers, 1, self.hiddenDim)))

	def forward(self, input):
		#print(joint_3d_vec.size())
		#x = joint_3d_vec
		#x = input.view(input.size()[0],1,input.size()[1])
		#print(x.size())
		#print(self.hidden[0].size(), self.hidden[1].size())
		lstm_out, self.hidden = self.lstm(input, self.hidden)
		y  = self.fullyConnected(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs

def train(model, num_epoch, batchSize = 5, lr=1e-3,rec_interval=50, disp_interval=25):
	global data, labels
	optimizer = optim.Adam(model.parameters(), lr)
	loss_values = []
	rec_step = 0
	for eph in range(num_epoch):
		print('epoch {} starting ...'.format(eph))
		avg_loss = 0
		n_samples = 0
		num_iter = len(data)//batchSize
		randpermed = torch.randperm(len(data))
		for i in range(num_iter):
			model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
			model.zero_grad()
			
			for k in range(batchSize):
				j = randpermed[i*batchSize + k]
				X= data[j].view(data[j].size()[0],1,75)
				Y = torch.LongTensor(1)
				Y[0]=labels[j]
				#print(X.size())
				n_samples += len(X)
				X = autograd.Variable(X)
				#print(X)
				Y = autograd.Variable(Y)
				y_hat = model(X)
				loss = F.cross_entropy(y_hat, Y)
				avg_loss += loss.data[0]
				if rec_step%rec_interval==0:
					loss_values.append(loss.data[0])
				loss.backward(retain_graph = True)
				rec_step += 1
			optimizer.step()
			if i % disp_interval == 0:
				print('epoch: %d iterations: %d loss :%g' % (eph, i, loss.data[0]))
		avg_loss /= n_samples
		#evaluating model accuracy
		#TrainAcc()
		print('epoch: {} <====train track===> avg_loss: {} \n'.format(eph, avg_loss))
	return loss_values






print("Loaded libraries")
data = getData()
print("Loaded training data")
labels = getLabels()
print("Loaded training labels")
valData = getValData()
print("Loaded validation data")
valLabels = getValLabels()
print("Loaded validation labels")
#labels = torch.from_numpy(labels).view(number,-1).type(torch.cuda.LongTensor)

#print(labels.size())

model = LSTMClassifier(label_size = 5)
TrainAcc()
ValAcc()
loss = train(model, 1)
#PlotLoss(loss)

