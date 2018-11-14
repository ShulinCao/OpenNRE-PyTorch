import numpy as np
import os
import json

in_path = "./raw_data/"
out_path = "./data"
case_sensitive = False
if not os.path.exists('./data'):
	os.mkdir('./data')
train_file_name = in_path + 'train.json'
test_file_name = in_path + 'test.json'
word_file_name = in_path + 'word_vec.json'
rel_file_name = in_path + 'rel2id.json'

def find_pos(sentence, head, tail):
	def find(sentence, entity):
		p = sentence.find(' ' + entity + ' ')
		if p == -1:
			if sentence[:len(entity) + 1] == entity + ' ':
				p = 0
			elif sentence[-len(entity) - 1:] == ' ' + entity:
				p = len(sentence) - len(entity)
			else:
				p = 0
		else:
			p += 1
		return p
		
	sentence = ' '.join(sentence.split())	
	p1 = find(sentence, head)
	p2 = find(sentence, tail)
	words = sentence.split()
	cur_pos = 0 
	pos1 = -1
	pos2 = -1
	for i, word in enumerate(words):
		if cur_pos == p1:
			pos1 = i
		if cur_pos == p2:
			pos2 = i
		cur_pos += len(word) + 1
	return pos1, pos2
		
def init(file_name, word_vec_file_name, rel2id_file_name, max_length = 120, case_sensitive = False, is_training = True):
	if file_name is None or not os.path.isfile(file_name):
		raise Exception("[ERROR] Data file doesn't exist")
	if word_vec_file_name is None or not os.path.isfile(word_vec_file_name):
		raise Exception("[ERROR] Word vector file doesn't exist")
	if rel2id_file_name is None or not os.path.isfile(rel2id_file_name):
		raise Exception("[ERROR] rel2id file doesn't exist")

	print("Loading data file...")
	ori_data = json.load(open(file_name, "r"))
	print("Finish loading")
	print("Loading word_vec file...")
	ori_word_vec = json.load(open(word_vec_file_name, "r"))
	print("Finish loading")
	print("Loading rel2id file...")
	rel2id = json.load(open(rel2id_file_name, "r"))
	print("Finish loading")
	
	if not case_sensitive:
		print("Eliminating case sensitive problem...")
		for i in ori_data:
			i['sentence'] = i['sentence'].lower()
			i['head']['word'] = i['head']['word'].lower()
			i['tail']['word'] = i['tail']['word'].lower()
		for i in ori_word_vec:
			i['word'] = i['word'].lower()
		print("Finish eliminating")
	
	# vec
	print("Building word vector matrix and mapping...")
	word2id = {}
	word_vec_mat = []
	word_size = len(ori_word_vec[0]['vec'])
	print("Got {} words of {} dims".format(len(ori_word_vec), word_size))
	for i in ori_word_vec:
		word2id[i['word']] = len(word2id)
		word_vec_mat.append(i['vec'])
	word2id['UNK'] = len(word2id)
	word2id['BLANK'] = len(word2id)
	word_vec_mat.append(np.random.normal(loc = 0, scale = 0.05, size = word_size))
	word_vec_mat.append(np.zeros(word_size, dtype = np.float32))
	word_vec_mat = np.array(word_vec_mat, dtype = np.float32)
	print("Finish building")
	
	# sorting
	print("Sorting data...")
	ori_data.sort(key = lambda a: a['head']['id'] + '#' + a['tail']['id'] + '#' + a['relation'])
	print("Finish sorting")
	
	sen_tot = len(ori_data)
	sen_word = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_pos1 = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_pos2 = np.zeros((sen_tot, max_length), dtype = np.int64)
	sen_mask = np.zeros((sen_tot, max_length, 3), dtype = np.float32)
	sen_label = np.zeros((sen_tot), dtype = np.int64)
	sen_len = np.zeros((sen_tot), dtype = np.int64)
	bag_label = []
	bag_scope = []
	bag_key = []
	for i in range(len(ori_data)):
		if  i%1000 == 0:
			print i
		sen = ori_data[i]
		# sen_label 
		if sen['relation'] in rel2id:
			sen_label[i] = rel2id[sen['relation']]
		else:
			sen_label[i] = rel2id['NA']
		words = sen['sentence'].split()
		# sen_len
		sen_len[i] = min(len(words), max_length)
		# sen_word
		for j, word in enumerate(words):
			if j < max_length:
				if word in word2id:
					sen_word[i][j] = word2id[word]
				else:
					sen_word[i][j] = word2id['UNK']
		for j in range(j + 1, max_length):
			sen_word[i][j] = word2id['BLANK']

		pos1, pos2 = find_pos(sen['sentence'], sen['head']['word'], sen['tail']['word'])
		if pos1 == -1 or pos2 == -1:
			raise Exception("[ERROR] Position error, index = {}, sentence = {}, head = {}, tail = {}".format(i, sen['sentence'], sen['head']['word'], sen['tail']['word']))
		if pos1 >= max_length:
			pos1 = max_length - 1
		if pos2 >= max_length:
			pos2 = max_length - 1
		pos_min = min(pos1, pos2)
		pos_max = max(pos1, pos2)
		for j in range(max_length):
			# sen_pos1, sen_pos2
			sen_pos1[i][j] = j - pos1 + max_length
			sen_pos2[i][j] = j - pos2 + max_length
			# sen_mask
			if j >= sen_len[i]:
				sen_mask[i][j] = [0, 0, 0]
			elif j - pos_min <= 0:
				sen_mask[i][j] = [100, 0, 0]
			elif j - pos_max <= 0:
				sen_mask[i][j] = [0, 100, 0]
			else:
				sen_mask[i][j] = [0, 0, 100]	
		# bag_scope	
		if is_training:
			tup = (sen['head']['id'], sen['tail']['id'], sen['relation'])
		else:
			tup = (sen['head']['id'], sen['tail']['id'])
		if bag_key == [] or bag_key[len(bag_key) - 1] != tup:
			bag_key.append(tup)
			bag_scope.append([i, i])
		bag_scope[len(bag_scope) - 1][1] = i

	print("Processing bag label...")
	# bag_label
	if is_training:
		for i in bag_scope:
			bag_label.append(sen_label[i[0]])
	else:
		for i in bag_scope:
			multi_hot = np.zeros(len(rel2id), dtype = np.int64)
			for j in range(i[0], i[1]+1):
				multi_hot[sen_label[j]] = 1
			bag_label.append(multi_hot)
	print("Finish processing")
	# ins_scope
	ins_scope = np.stack([list(range(len(ori_data))), list(range(len(ori_data)))], axis = 1)
	print("Processing instance label...")
	# ins_label
	if is_training:
		ins_label = sen_label
	else:
		ins_label = []
		for i in sen_label:
			one_hot = np.zeros(len(rel2id), dtype = np.int64)
			one_hot[i] = 1
			ins_label.append(one_hot)
		ins_label = np.array(ins_label, dtype = np.int64)
	print("Finishing processing")
	bag_scope = np.array(bag_scope, dtype = np.int64)
	bag_label = np.array(bag_label, dtype = np.int64)
	ins_scope = np.array(ins_scope, dtype = np.int64)
	ins_label = np.array(ins_label, dtype = np.int64)
	
	# saving
	print("Saving files")
	if is_training:
		name_prefix = "train"
	else:
		name_prefix = "test"
	np.save(os.path.join(out_path, 'vec.npy'), word_vec_mat)
	np.save(os.path.join(out_path, name_prefix + '_word.npy'), sen_word)
	np.save(os.path.join(out_path, name_prefix + '_pos1.npy'), sen_pos1)
	np.save(os.path.join(out_path, name_prefix + '_pos2.npy'), sen_pos2)
	np.save(os.path.join(out_path, name_prefix + '_mask.npy'), sen_mask)
	np.save(os.path.join(out_path, name_prefix + '_bag_label.npy'), bag_label)
	np.save(os.path.join(out_path, name_prefix + '_bag_scope.npy'), bag_scope)
	np.save(os.path.join(out_path, name_prefix + '_ins_label.npy'), ins_label)
	np.save(os.path.join(out_path, name_prefix + '_ins_scope.npy'), ins_scope)
	print("Finish saving")		

init(train_file_name, word_file_name, rel_file_name, max_length = 120, case_sensitive = False, is_training = True)
init(test_file_name, word_file_name, rel_file_name, max_length = 120, case_sensitive = False, is_training = False)
