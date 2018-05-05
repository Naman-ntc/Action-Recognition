import torch.nn as nn
from torch import autograd
from torch import optim
import torch.nn.functional as F

 model = torch.load('hgreg-3d.pth').cuda()


class CombinedLSTM(nn.Module):
	"""docstring for CombinedLSTM"""
	def __init__(self, inHeight, inWidth, inChannels, featHidden, skeHidden, outDim):
		super(CombinedLSTM, self).__init__()
		self.inHeight = inHeight
		self.inWidth = inWidth
		self.inChannels = inChannels
		self.featHidden = featHidden
		self.skeHidden = skeHidden
		self.outDim = outDim

	def forward(self, input):
		output = model(input)
 		pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
 		reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
 		point_3d = np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1)
