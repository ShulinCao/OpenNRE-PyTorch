#coding:utf-8
import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import datetime
import ctypes
import json
import sys
import sklearn.metrics

def to_var(x):
    return Variable(torch.from_numpy(x).cuda())

class Accuracy(object):

    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class Config(object):

    def __init__(self):
	self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.export_path = './data'
        self.max_length = 120
        self.pos_num = 100
        self.num_classes = 53
        self.hidden_size = 230
        self.pos_size = 5
        self.word_size = 50
        self.max_epoch = 60
        self.batch_size = 160
        self.opt_method = 'SGD'
        self.optimizer = None
        self.learning_rate = 0.5
        self.weight_decay = 1e-5
        self.drop_prob = 0.5
        self.checkpoint_dir = './checkpoint'
        self.test_result_dir = './test_result'
        self.save_epoch = 1
        self.pretrain_model = 'None'
        self.is_training = True
        self.use_bag = True
        self.trainModel = None
        self.gpu = True

    def set_export_path(self, export_path):
    	self.export_path = export_path

    def set_max_length(self, max_length):
    	self.max_length = max_length

    def set_pos_num(self, pos_num):
    	self.pos_num = pos_num

    def set_num_classes(self, num_classes):
    	self.num_classes = num_classes

    def set_hidden_size(self, hidden_size):
    	self.hidden_size = hidden_size

    def set_window_size(self, window_size):
        self.window_size = window_size
        
    def set_pos_size(self, pos_size):
    	self.pos_size = pos_size

    def set_word_size(self, word_size):
    	self.word_size = word_size

    def set_max_epoch(self, max_epoch):
    	self.max_epoch = max_epoch

    def set_batch_size(self, batch_size):
    	self.batch_size = batch_size

    def set_opt_method(self, opt_method):
    	self.opt_method = opt_method

    def set_learning_rate(self, learning_rate):
    	self.learning_rate = learning_rate

    def set_weight_decay(self, weight_decay):
    	self.weight_decay = weight_decay

    def set_drop_prob(self, drop_prob):
    	self.drop_prob = drop_prob

    def set_checkpoint_dir(self, checkpoint_dir):
    	self.checkpoint_dir = checkpoint_dir

    def set_test_result_dir(self, test_result_dir):
    	self.test_result_dir = test_result_dir

    def set_save_epoch(self, save_epoch):
    	self.save_epoch = save_epoch

    def set_pretrain_model(self, pretrain_model):
    	self.pretrain_model = pretrain_model

    def set_is_training(self, is_training):
    	self.is_training = is_training

    def set_use_bag(self, use_bag):
    	self.use_bag = use_bag

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range
    
    def set_gpu(self, True):
        self.gpu = gpu

    def load_train_data(self):
    	print('reading training data...')
        self.data_word_vec = np.load(os.path.join(self.export_path, 'vec.npy'))
        self.data_instance_triple = np.load(os.path.join(self.export_path, 'train_instance_triple.npy'))
        self.data_instance_scope = np.load(os.path.join(self.export_path, 'train_instance_scope.npy'))
	self.data_instance_label = np.load(os.path.join(self.export_path, 'train_instance_label.npy'))
        self.data_train_length = np.load(os.path.join(self.export_path, 'train_len.npy'))
        self.data_train_label = np.load(os.path.join(self.export_path, 'train_label.npy'))
        self.data_train_word = np.load(os.path.join(self.export_path, 'train_word.npy'))
        self.data_train_pos1 = np.load(os.path.join(self.export_path, 'train_pos1.npy'))
        self.data_train_pos2 = np.load(os.path.join(self.export_path, 'train_pos2.npy'))
        self.data_train_mask = np.load(os.path.join(self.export_path, 'train_mask.npy'))

        print('reading finished')
        print('mentions         : %d' % (len(self.data_instance_triple)))
        print('sentences        : %d' % (len(self.data_train_length)))
        print('relations        : %d' % (self.num_classes))
        print('word size        : %d' % (self.word_size))
        print('position size     : %d' % (self.pos_size))
        print('hidden size        : %d' % (self.hidden_size))

    def load_test_data(self):
        print('reading test data...')
        self.data_word_vec = np.load(os.path.join(self.export_path, 'vec.npy'))
        self.data_instance_entity = np.load(os.path.join(self.export_path, 'test_instance_entity.npy'))
        self.data_instance_entity_no_bag = np.load(os.path.join(self.export_path, 'test_instance_entity_no_bag.npy'))
        instance_triple = np.load(os.path.join(self.export_path, 'test_instance_triple.npy'))
        self.data_instance_triple = {}
        for item in instance_triple:
            self.data_instance_triple[(item[0], item[1], int(item[2]))] = 0
        self.data_instance_scope = np.load(os.path.join(self.export_path, 'test_instance_scope.npy'))
        self.data_test_length = np.load(os.path.join(self.export_path, 'test_len.npy'))
        self.data_test_label = np.load(os.path.join(self.export_path, 'test_label.npy'))
        self.data_test_word = np.load(os.path.join(self.export_path, 'test_word.npy'))
        self.data_test_pos1 = np.load(os.path.join(self.export_path, 'test_pos1.npy'))
        self.data_test_pos2 = np.load(os.path.join(self.export_path, 'test_pos2.npy'))
        self.data_test_mask = np.load(os.path.join(self.export_path, 'test_mask.npy'))

        print('reading finished')
        print('mentions         : %d' % (len(self.data_instance_triple)))
        print('sentences        : %d' % (len(self.data_test_length)))
        print('relations        : %d' % (self.num_classes))
        print('word size        : %d' % (self.word_size))
        print('position size     : %d' % (self.pos_size))
        print('hidden size        : %d' % (self.hidden_size))    

    def init(self):
    	if self.is_training:
    	    self.load_train_data()
    	    if self.use_bag:
    	        self.train_order = list(range(len(self.data_instance_triple)))
    	    else:
    	        self.train_order = list(range(len(self.data_train_word)))
            self.nbatches = int(len(self.data_instance_triple) / float(self.batch_size))
    	else:
    	    self.load_test_data()
            self.nbatches = int(len(self.data_instance_scope) / self.batch_size)

    def set_train_model(self, model):
    	print('initializing training model...')
        self.model = model
        self.trainModel = self.model(config = self)
        if self.pretrain_model != "None":
            self.trainModel.load_state_dict(torch.load(self.pretrain_model))
        self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.learning_rate, lr_decay=self.lr_decay, weight_decay=self.weight_decay)
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        print('initializing finished')

    def set_test_model(self, model):
        print('initializing test model...')
        self.model = model
        self.testModel = self.model(config = self)
        self.testModel.cuda()
        self.testModel.eval()
        print('initializing finished')

    def sampling(self, batch):
    	if self.use_bag:
    	    input_scope = np.take(self.data_instance_scope, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
	    input_label = np.take(self.data_instance_label, self.train_order[batch * self.batch_size : (batch + 1) * self.batch_size], axis = 0)
    	    index = []
            scope = [0]
            weights = []
            for num in input_scope:
                index = index + list(range(num[0], num[1] + 1))
                scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_train_word[index, :]
        self.batch_pos1 = self.data_train_pos1[index, :]
        self.batch_pos2 = self.data_train_pos2[index, :]
        self.batch_mask = self.data_train_mask[index, :, :]
        self.batch_length = self.data_train_length[index]
        self.batch_label = input_label
        self.batch_attention_query = self.data_train_label[index]
        self.batch_scope = scope

    def get_test_batch(self, batch):
        input_scope = self.data_instance_scope[batch * self.batch_size:min((batch + 1) * self.batch_size, len(self.data_instance_scope))]
        index = []
        scope = [0]
        for num in input_scope:
            index = index + list(range(num[0], num[1] + 1))
            scope.append(scope[len(scope) - 1] + num[1] - num[0] + 1)
        self.batch_word = self.data_test_word[index, :]
        self.batch_pos1 = self.data_test_pos1[index, :]
        self.batch_pos2 = self.data_test_pos2[index, :]
        self.batch_mask = self.data_test_mask[index, :, :]
        self.batch_length = self.data_test_length[index]
        self.batch_scope = scope

    def train_one_step(self):
    	self.trainModel.embedding.word = to_var(self.batch_word)
    	self.trainModel.embedding.pos1 = to_var(self.batch_pos1)
    	self.trainModel.embedding.pos2 = to_var(self.batch_pos2)
    	self.trainModel.encoder.mask = to_var(self.batch_mask)
    	self.trainModel.encoder.length = to_var(self.batch_length)
    	self.trainModel.selector.scope = self.batch_scope
    	self.trainModel.selector.attention_query = to_var(self.batch_attention_query)
    	self.trainModel.classifier.label = to_var(self.batch_label)
        self.optimizer.zero_grad()
    	loss , _output = self.trainModel()
        loss.backward()
        self.optimizer.step()
    	for i, prediction in enumerate(_output):
            if self.batch_label[i] == 0:
                self.acc_NA.add(prediction == self.batch_label[i])
            else:
                self.acc_not_NA.add(prediction == self.batch_label[i])
            self.acc_total.add(prediction == self.batch_label[i])
        return loss.data[0]

    def test_one_step(self):
        self.testModel.embedding.word = to_var(self.batch_word)
        self.testModel.embedding.pos1 = to_var(self.batch_pos1)
        self.testModel.embedding.pos2 = to_var(self.batch_pos2)
        self.testModel.encoder.mask = to_var(self.batch_mask)
        self.testModel.encoder.length = to_var(self.batch_length)
        self.testModel.selector.scope = self.batch_scope
        return self.testModel.test()


    def train(self):
    	if not os.path.exists(self.checkpoint_dir):
    	    os.mkdir(self.checkpoint_dir)
    	for epoch in range(self.max_epoch):
    	    print('epoch '+ str(epoch) + ' starts...')
    	    self.acc_NA.clear()
    	    self.acc_not_NA.clear()
    	    self.acc_total.clear()
    	    np.random.shuffle(self.train_order)
    	    for batch in range(self.nbatches):
    	    	self.sampling(batch)
    		loss = self.train_one_step()
                time_str = datetime.datetime.now().isoformat()
                sys.stdout.write("epoch %d step %d time %s | loss : %f, NA accuracy: %f, not NA accuracy: %f, total accuracy %f" % (epoch, batch, time_str, loss, self.acc_NA.get(), self.acc_not_NA.get(), self.acc_total.get()) + '\n')
                sys.stdout.flush()
            if (epoch + 1) % self.save_epoch == 0:
                print('epoch ' + str(epoch + 1) + ' has finished')
                print('saving model...')
                path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
                torch.save(self.trainModel.state_dict(), path)
                print('have saved model to ' + path)

    def test(self):
        epoch_range = eval(self.epoch_range)
        epoch_range = range(epoch_range[0], epoch_range[1])
        save_x = None
        save_y = None
        best_auc = 0
        best_epoch = 0
        print('test ' + self.model.__name__)
        for epoch in epoch_range:
            path = os.path.join(self.checkpoint_dir, self.model.__name__ + '-' + str(epoch))
            if not os.path.exists(path):
                continue
            print('start testing checkpoint, iteration =', epoch)
            self.testModel.load_state_dict(torch.load(path))
            stack_output = []
            stack_label = []
            test_result = []
            total_recall = 0 
            for batch in range(self.nbatches):
                self.get_test_batch(batch)
                self.test_output = self.test_one_step()
                for i in range(len(self.test_output)):
                    pred = self.test_output[i]
                    entity = self.data_instance_entity[i + batch * self.batch_size]
                    for rel in range(1, len(pred)):
                        flag = int(((entity[0], entity[1], rel) in self.data_instance_triple))
                        total_recall += flag
                        test_result.append([(entity[0], entity[1], rel), flag, pred[rel]])

                if batch % 100 == 0:
                    sys.stdout.write('predicting {} / {}\n'.format(batch, self.nbatches))
                    sys.stdout.flush()
            
            print('\nevaluating...')

            sorted_test_result = sorted(test_result, key=lambda x: x[2])
            pr_result_x = []
            pr_result_y = []
            correct = 0
            for i, item in enumerate(sorted_test_result[::-1]):
                if item[1] == 1:
                    correct += 1
                pr_result_y.append(float(correct) / (i + 1))
                pr_result_x.append(float(correct) / total_recall)
            auc = sklearn.metrics.auc(x=pr_result_x, y=pr_result_y)
            print('auc:', auc)
            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                save_x = pr_result_x
                save_y = pr_result_y

        if not os.path.exists(self.test_result_dir):
            os.mkdir(self.test_result_dir)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_x.npy'), save_x)
        np.save(os.path.join(self.test_result_dir, self.model.__name__ + '_y.npy'), save_y)
        print('best epoch:', best_epoch)
