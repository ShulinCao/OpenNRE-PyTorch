import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='pcnn_att', help='name of the model')
args = parser.parse_args()
model = {
	'pcnn_att': models.PCNN_ATT,
	'pcnn_max': models.PCNN_MAX,
	'pcnn_ave': models.PCNN_AVE,
	'cnn_att' : models.CNN_ATT,
	'cnn_ave' : models.CNN_AVE,
	'cnn_max' : models.CNN_MAX
}
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
param_file = open(os.path.join('data', "config"), 'r')
param = json.loads(param_file.read())
param_file.close()

con = config.Config()
con.set_export_path('./data')
con.set_max_length(param['fixlen'])
con.set_pos_num(param['maxlen'] * 2 + 1)
con.set_num_classes(len(param['relation2id']))
con.set_hidden_size(230)
con.set_window_size(3)
con.set_pos_size(5)
con.set_word_size(50)
con.set_max_epoch(20)
con.set_batch_size(160)
con.set_opt_method('SGD')
con.set_learning_rate(0.5)
con.set_weight_decay(1e-5)
con.set_drop_prob(0.5)
con.set_checkpoint_dir('./checkpoint/')
con.set_test_result_dir('./test_result')
con.set_save_epoch(1)
con.set_pretrain_model('None')
con.set_is_training(False)
con.set_use_bag(True)
con.set_epoch_range('0,20')
con.init()
con.set_test_model(model[args.model_name])
con.test()
