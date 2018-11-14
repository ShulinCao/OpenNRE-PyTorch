import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Selector(nn.Module):
	def __init__(self, config, relation_dim):
		super(Selector, self).__init__()
		self.config = config
		self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)
		self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
		self.attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)
		self.init_weights()
		self.scope = None
		self.attention_query = None
		self.label = None
		self.dropout = nn.Dropout(self.config.drop_prob)
	def init_weights(self):	
		nn.init.xavier_uniform(self.relation_matrix.weight.data)
		nn.init.normal(self.bias)
		nn.init.xavier_uniform(self.attention_matrix.weight.data)
	def get_logits(self, x):
		logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1),) + self.bias
		return logits
	def forward(self, x):
		raise NotImplementedError
	def test(self, x):
		raise NotImplementedError

class Attention(Selector):
	def _attention_train_logit(self, x):
		relation_query = self.relation_matrix(self.attention_query)
		attention = self.attention_matrix(self.attention_query)
		attention_logit = torch.sum(x * attention * relation_query, 1, True)
		return attention_logit
	def _attention_test_logit(self, x):
		attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0, 1))
		return attention_logit
	def forward(self, x):
		attention_logit = self._attention_train_logit(x)
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return logits
	def test(self, x):
		attention_logit = self._attention_test_logit(x)	
		tower_output = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i] : self.scope[i + 1]], 0, 1), 1)
			final_repre = torch.matmul(attention_score, sen_matrix)
			logits = self.get_logits(final_repre)
			tower_output.append(torch.diag(F.softmax(logits, 1)))
		stack_output = torch.stack(tower_output)
		return list(stack_output.data.cpu().numpy())

class One(Selector):
	def forward(self, x):
		tower_logits = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			sen_matrix = self.dropout(sen_matrix)
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			_, k = torch.max(score, dim = 0)
			k = k[self.label[i]]
			tower_logits.append(logits[k])
		return torch.cat(tower_logits, 0)
	def test(self, x):
		tower_score = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			logits = self.get_logits(sen_matrix)
			score = F.softmax(logits, 1)
			score, _ = torch.max(score, 0)
			tower_score.append(score)
		tower_score = torch.stack(tower_score)
		return list(tower_score.data.cpu().numpy())

class Average(Selector):
	def forward(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i+ 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		stack_repre = self.dropout(stack_repre)
		logits = self.get_logits(stack_repre)
		return logits
	def test(self, x):
		tower_repre = []
		for i in range(len(self.scope) - 1):
			sen_matrix = x[self.scope[i] : self.scope[i + 1]]
			final_repre = torch.mean(sen_matrix, 0)
			tower_repre.append(final_repre)
		stack_repre = torch.stack(tower_repre)
		logits = self.get_logits(stack_repre)
		score = F.softmax(logits, 1)
		return list(score.data.cpu().numpy())
