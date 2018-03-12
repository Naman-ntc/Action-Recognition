import torch
from helperFunctions import *

def TrainAcc():
	print(checkAcc(model,data,labels, length = 1000))

def ValAcc():
	print(checkAcc(model, valData, valLabels))

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
		return (autograd.Variable(torch.zeros(self.layers, 1, self.hidden_dim)),
				autograd.Variable(torch.zeros(self.layers, 1, self.hidden_dim)))

	def forward(self, input):
		#print(joint_3d_vec.size())
		#x = joint_3d_vec
		x = x.view(x.size()[0],1,x.size()[1])
		#print(x.size())
		#print(self.hidden[0].size(), self.hidden[1].size())
		lstm_out, self.hidden = self.lstm(x, self.hidden)
		y  = self.fullyConnected(lstm_out[-1])
		log_probs = F.log_softmax(y)
		return log_probs








data = getData()
labels = getLabels()
valData = getValData()
valLabels = getValLabels()
labels = torch.from_numpy(neededLabels).view(number,-1).type(torch.cuda.LongTensor)
