import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class _CNN(nn.Module):
    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_length
        self.in_width = self.config.word_size + 2 * self.config.pos_size
        self.kernel_size = (self.config.window_size, self.in_width)
        self.out_channels = self.config.hidden_size
        self.stride = (1, 1)
        self.padding = (1, 0)
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
    def forward(self, embedding):
        return self.cnn(embedding)

class _PiecePooling(nn.Module):
    def __init__(self):
        super(_PiecePooling, self).__init__()
    def forward(self, x, mask, hidden_size):
        mask = torch.unsqueeze(mask, 1)
        x, _ = torch.max(mask + x, dim = 2)
        x = x - 100
        return x.view(-1, hidden_size * 3) 

class _MaxPooling(nn.Module):
    def __init__(self):
        super(_MaxPooling, self).__init__()
    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim = 2)
        return x.view(-1, hidden_size)

class PCNN(nn.Module):
    def __init__(self, config):
        super(PCNN, self).__init__()
        self.config = config
        self.length = None
        self.mask = None
        self.cnn = _CNN(config)
        self.pooling = _PiecePooling()
	self.activation = nn.ReLU()
    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim = 1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.mask, self.config.hidden_size)
	return self.activation(x)

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.length = None
        self.mask = None
        self.cnn = _CNN(config)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()
    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim = 1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.config.hidden_size)
        return self.activation(x)

