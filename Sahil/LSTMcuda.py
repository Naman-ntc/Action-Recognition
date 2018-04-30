import torch
from cudaHelperFunctions import *
import torch.nn as nn
from torch import autograd
from torch import optim
import torch.nn.functional as F
import pickle

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def TrainAcc(l = 500):
	print("The training accuracy is:", )
	print(checkAcc(model,data,labels, length = l))

def ValAcc(l = 500):
	print("The validation accuracy is:",)
	print(checkAcc(model, valData, valLabels, length = l))

class LSTMClassifier(nn.Module):

	def __init__(self, hidden_dim=160, label_size=8, input_dim=48, num_layers = 2):
		super(LSTMClassifier, self).__init__()
		self.hiddenDim = hidden_dim
		self.layers = num_layers
		self.embedding = nn.Linear(input_dim, 32)
		self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_dim, num_layers = num_layers)
		self.fullyConnected = nn.Linear(hidden_dim, label_size)
		self.hidden = self.init_hidden()
		
	def init_hidden(self):
		# the first is the hidden h
		# the second is the cell  c
		return (autograd.Variable(torch.zeros(self.layers, 1, self.hiddenDim).type(torch.cuda.FloatTensor)),
				autograd.Variable(torch.zeros(self.layers, 1, self.hiddenDim).type(torch.cuda.FloatTensor)))

	def forward(self, input):
		#print(joint_3d_vec.size())
		#x = joint_3d_vec
		#x = input.view(input.size()[0],1,input.size()[1])
		#print(x.size())
		#print(self.hidden[0].size(), self.hidden[1].size())
		#print(type(input))
		#print(input.size())
		#print(input.type())
		x = autograd.Variable(input)
		#x = self.embedding(input.view(input.size()[0], 75))
		x = self.embedding(x)
		#print(x.size())
		#print(x.view(x.size()[0], 1, 64).size())
		#print(type(x), type(self.hidden[0]))
		lstm_out, self.hidden = self.lstm(x.view(x.size()[0],1, 32), self.hidden)
		y  = self.fullyConnected(lstm_out[-1])
		log_probs = F.log_softmax(y, dim=1)
		return log_probs

avg_loss_values = []
loss_values = []
trainAccuracies = []
valAccuracies = []
bestAcc = 0.0

def train(model, num_epoch, epoch_size = -1, batchSize = 5, lr=1e-3,rec_interval=10, disp_interval=20):
	global data, labels, valData, valLabels, bestAcc
	optimizer = optim.Adam(model.parameters(), lr)
	global avg_loss_values, loss_values, trainAccuracies, valAccuracies
	rec_step = 0
	for eph in range(num_epoch):
		print('epoch {} starting ...'.format(eph))
		avg_loss = 0
		n_samples = 0
		totalSamples = len(data)
		if epoch_size == -1:
			num_iter = len(data)//batchSize
		else:
			num_iter = epoch_size//batchSize
		randpermed = torch.randperm(len(data))
		for i in range(1, num_iter):
			model.hidden = (model.hidden[0].detach(), model.hidden[1].detach())
			model.zero_grad()
			totalLoss = 0.0
			
			for k in range(batchSize):
				j = randpermed[i*batchSize + k]
				X= data[j]
				Y = torch.cuda.LongTensor(1)
				Y[0]=labels[j]
				#print(X.size())
				n_samples += len(X)
				#print(X)
				Y = autograd.Variable(Y)
				y_hat = model(X)
				loss = F.cross_entropy(y_hat, Y)
				avg_loss += loss.data[0]
				totalLoss += loss.data[0]
				loss.backward(retain_graph = True)
				rec_step += 1
			optimizer.step()
			if i % disp_interval == 0:
				if i*batchSize < totalSamples - 60:
					#print(i)
					trainAccuracies.append(checkAcc(model, data, labels, start = batchSize*i, length = 50))
					valAccuracies.append(checkAcc(model, valData, valLabels, start = (batchSize*i) // 3, length = 50))
				print('epoch: %d iterations: %d loss: %g trainAcc: %g valAcc: %g' % (eph, i, avg_loss/(10.0*batchSize), trainAccuracies[-1], valAccuracies[-1]))
			if rec_step%rec_interval==0:
				loss_values.append(totalLoss/batchSize)
				avg_loss/=10.0
				avg_loss_values.append(avg_loss/batchSize)
				avg_loss = 0.0
		
		avg_loss /= 10
		#evaluating model accuracy
		#TrainAcc()
		print('epoch: {} <====train track===> avg_loss: {} \n'.format(eph, avg_loss))
		
		torch.save(model, open("currentModel.pth", 'wb'))
		currentAcc = checkAcc(model, valData, valLabels, start = 0, length = -1)
		print("The val accuracy is", currentAcc)
		if currentAcc > bestAcc:
			bestAcc = currentAcc
			torch.save(model, open("bestModel.pth", 'wb'))


	PlotLoss(loss_values, name = 'oneLSTMloss.png')
	return loss_values


def storeData():
	masterList = [avg_loss_values, loss_values, trainAccuracies, valAccuracies]
	pickle.dump(masterList, open("fourLSTMlog.npy", 'wb'))

def loadData():
	masterList = pickle.load(open("fourLSTMlog.npy", 'rb'))
	global avg_loss_values, loss_values, trainAccuracies, valAccuracies
	avg_loss_values = masterList[0]
	loss_values = masterList[1]
	trainAccuracies = masterList[2]
	valAccuracies = masterList[3]

def plotData():
	PlotLoss(avg_loss_values, "avg_loss_values.png")
	PlotLoss(loss_values, "loss_values.png")
	PlotAccuracies(trainAccuracies, valAccuracies, "accuracies.png")

print("Loaded libraries")
data = getData()
print("Loaded training data")
labels = getLabels()
print("Loaded training labels")
valData = getValData()
print("Loaded validation data")
valLabels = getValLabels()
print("Loaded validation labels")
#labels = labels.view(number,-1).type(torch.cuda.LongTensor)

#print(labels.size())

model = LSTMClassifier(hidden_dim = 128, num_layers = 2, label_size = 8)
#PlotLoss(loss)
