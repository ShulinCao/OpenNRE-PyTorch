import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Classifier(nn.Module):
	def __init__(self, config):
		super(Classifier, self).__init__()
		self.config = config
		self.label = None
		self.loss = nn.CrossEntropyLoss()
	def forward(self, logits):
		loss = self.loss(logits, self.label)
		_, output = torch.max(logits, dim = 1)
		return loss, output.data
