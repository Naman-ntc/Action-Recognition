import torch
from cudaHelperFunctions import *
import torch.nn as nn
from torch import autograd
from torch import optim
import torch.nn.functional as F

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def TrainAcc(l = 500):
	print("The training accuracy is:", )
	print(checkAcc(model,data,labels, length = l)[0])

def ValAcc():
	print("The validation accuracy is:",)
	print(checkAcc(model, valData, valLabels)[0])

class LSTMClassifier(nn.Module):

	def __init__(self, hidden_dim=128, label_size=49, input_dim=75, num_layers = 1):
		super(LSTMClassifier, self).__init__()
		self.hiddenDim = hidden_dim
		self.layers = num_layers
		self.embedding = nn.Linear(input_dim, 64)
		self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers = num_layers)
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
		lstm_out, self.hidden = self.lstm(x.view(x.size()[0],1, 64), self.hidden)
		y  = self.fullyConnected(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs

def train(model, num_epoch, epoch_size = -1, batchSize = 5, lr=1e-3,rec_interval=5, disp_interval=1):
	global data, labels
	optimizer = optim.Adam(model.parameters(), lr)
	loss_values = []
	rec_step = 0
	for eph in range(num_epoch):
		print('epoch {} starting ...'.format(eph))
		avg_loss = 0
		n_samples = 0
		if epoch_size == -1:
			num_iter = len(data)//batchSize
		else:
			num_iter = epoch_size//batchSize
		randpermed = torch.randperm(len(data))
		for i in range(num_iter):
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
				loss.backward(retain_variables = True)
				rec_step += 1
			optimizer.step()
			if i % disp_interval == 0:
				print('epoch: %d iterations: %d loss :%g' % (eph, i, totalLoss/batchSize))
			if rec_step%rec_interval==0:
				loss_values.append(totalLoss/batchSize)
				
		avg_loss /= n_samples
		#evaluating model accuracy
		#TrainAcc()
		print('epoch: {} <====train track===> avg_loss: {} \n'.format(eph, avg_loss))
	PlotLoss(loss_values, name = 'oneLSTMloss.png')
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
#PlotLoss(loss)



def Scheduler():
	loss0 = []
	loss1 = []
	loss2 = []
	loss3 = []
	loss4 = []
	loss5 = []
	#PlotLoss(,'loss1.png')
	contin = bool(input("Do you want to continue training: "))
	TrainAcc()
	ValAcc()
	if contin:
		loss0 = train(model,5,batchSize = 16, lr = 1e-4)
	TrainAcc()
	ValAcc()
	contin = bool(input("Do you want to continue training: "))
	if contin:
		loss1 = train(model,10,batchSize = 16, lr = 3e-5)
	#PlotLoss(loss1,'loss1.png')
	TrainAcc()
	ValAcc()
	contin = bool(input("Do you want to continue training: "))
	if contin:
		loss2 = train(model,10,batchSize = 16,lr = 1e-5)
	TrainAcc()
	ValAcc()
	contin = bool(input("Do you want to continue training: "))
	if contin:
		loss3 = train(model,10,batchSize = 16,lr=5e-6)
	#PlotLoss(loss1+loss2+loss3,'loss2.png')
	TrainAcc()
	ValAcc()
	contin = bool(input("Do you want to continue training: "))
	if contin:
		loss4 = train(model,20,batchSize = 8,lr = 5e-6)
	print(checkAcc(model,data,labels, length = -1)[0])
	ValAcc()
	PlotLoss(loss0 + loss1+loss2+loss3+loss4+loss5)
	#loss5 = train(model0,50,3300,1e-5)
	#PlotLoss(loss1+loss2+loss3+loss4+loss5,'loss3.png')
	#TrainAcc()


