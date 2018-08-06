import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config =config
        self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
        self.pos1_embedding = nn.Embedding(self.config.pos_num + 1, self.config.pos_size, padding_idx = self.config.pos_num)
        self.pos2_embedding = nn.Embedding(self.config.pos_num + 1, self.config.pos_size, padding_idx = self.config.pos_num)
        self.init_word_weights()
        self.init_pos_weights()
        self.word = None
        self.pos1 = None
        self.pos2 = None

    def init_word_weights(self):
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))

    def init_pos_weights(self):
        nn.init.xavier_uniform(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self):
        word = self.word_embedding(self.word)
        pos1 = self.pos1_embedding(self.pos1)
        pos2 = self.pos2_embedding(self.pos2)
        embedding = torch.cat((word, pos1, pos2), dim = 2)
        return embedding
