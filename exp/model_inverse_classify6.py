import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData
import sys
import matplotlib.pyplot as plt

model_path = 'models/inverse_classify.mdl'
model_path_norm = model_path + '.norm'

counter = 0
def test_valid(obs1, obs2):
	global counter
	if np.count_nonzero(obs1 == -1)==14 and np.count_nonzero(obs2 == -1)==14:
		return 0
	else:
		return 1
	
	
def test_action(act):
	if np.array_equal(act, [0, 0, 0]): return 0
	if np.array_equal(act, [0.5, 0, 0]): return 1
	if np.array_equal(act, [-0.5, 0, 0]): return 2
	if np.array_equal(act, [0, 0.5, 0]): return 3
	if np.array_equal(act, [0, -0.5, 0]): return 4
	if np.array_equal(act, [0, 0, 0.2]): return 5
	if np.array_equal(act, [0, 0, -0.2]): return 6

def get_sets(data):
	datas = []
	sets = []
	acts = None
	for d in data:
		if not np.array_equal(acts, d[2]):
			acts = d[2]	
			cursets  = sets
			sets = []
			datas.append(cursets)
		sets.append(d)
	cursets  = sets
	datas.append(cursets)
	datas = datas[1:]
	s = datas[0]
	return datas

def get_newdatas(datas):
	new_datas = []
	for new_data in datas:
		for i in range(len(new_data)):
			for j in range(i+1,len(new_data)):
				new_datas.append((new_data[i],new_data[j]))
	return new_datas

class LinearNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		hid_dim = int(sys.argv[1])
		self.linear11 = torch.nn.Linear(14, hid_dim)
		self.linear21 = torch.nn.Linear(hid_dim, hid_dim)

		self.linear12 = torch.nn.Linear(14, hid_dim)
		self.linear22 = torch.nn.Linear(hid_dim, hid_dim)
		
		self.linear3 = torch.nn.Linear(2*hid_dim, 7)

	def forward_obs1(self, obs):
		res = F.relu(self.linear11(obs))
		res = F.relu(self.linear21(res))
		return res

	def forward_obs2(self, obs):
		res = F.relu(self.linear12(obs))
		res = F.relu(self.linear22(res))
		return res

		
	def forward(self, obs1, obs2):
		"""
		obs1 = [r for r in torch.split(obs1, 2, dim=1)]
		obs1 = torch.stack(obs1, dim=1)
	
		obs2 = [r for r in torch.split(obs2, 2, dim=1)]
		obs2 = torch.stack(obs2, dim=1)
		"""
		"""
		obs1 = self.forward_obs(obs1)
		obs2 = self.forward_obs(obs2)
		obs = torch.cat([obs1, obs2-obs1], dim=1) #(32, 6, 4)
		"""
		x = self.forward_obs1(obs1)		
		y = self.forward_obs2(obs2-obs1)
		obs = torch.cat([x, y], dim=1)
		pred = self.linear3(obs) #(32, 6, 7)
		# pred = pred.sum(dim=1)
	
		return pred

def train():
	# source = 'sim'
	source = 'real'
	data_size = 20000
	all_obj = True
	# all_obj = False
	
	save_model = False
	# save_model = True
	
	skipdata = 1
	no_zerobeacon = False
	#####################################################################
	
	def test_action(act):
		if np.array_equal(act, [0, 0, 0]): return 0
		if np.array_equal(act, [0.5, 0, 0]): return 1
		if np.array_equal(act, [-0.5, 0, 0]): return 2
		if np.array_equal(act, [0, 0.5, 0]): return 3
		if np.array_equal(act, [0, -0.5, 0]): return 4
		if np.array_equal(act, [0, 0, 0.2]): return 5
		if np.array_equal(act, [0, 0, -0.2]): return 6        
	data_test = RealData(all_obj=all_obj, path='note_test.txt')
	
	
	K = 1
	"""
	data_test = [(
		data_test[i][0], 
		data_test[i][1], 
		test_action(data_test[i][2]), 
		data_test[i+1][1] - data_test[i][1],
	) for i in range(K, len(data_test)-1)]
	"""
	data_test = [(
		data_test[i][0], 
		data_test[i][1], 
		test_action(data_test[i][2]), 
		data_test[i+1][1], #- data_test[i][1],
		test_valid(data_test[i][1], data_test[i+1][1])
	) for i in range(K, len(data_test)-1)]
	print(data_test[0])
	#####################################################################
	
	
	if source == 'sim':
		def discrete_action(act):
			if np.array_equal(act, [0, 0, 0]): return 0
			if np.array_equal(act, [0.5, 0, 0]): return 1
			if np.array_equal(act, [-0.5, 0, 0]): return 2
			if np.array_equal(act, [0, 0.5, 0]): return 3
			if np.array_equal(act, [0, -0.5, 0]): return 4
			if np.array_equal(act, [0, 0, 0.5]): return 5
			if np.array_equal(act, [0, 0, -0.5]): return 6
	
		data = SimData(data_size, all_obj=all_obj)
		
	else:
		def discrete_action(act):
			if np.array_equal(act, [0, 0, 0]): return 0
			if np.array_equal(act, [0.5, 0, 0]): return 1
			if np.array_equal(act, [-0.5, 0, 0]): return 2
			if np.array_equal(act, [0, 0.5, 0]): return 3
			if np.array_equal(act, [0, -0.5, 0]): return 4
			if np.array_equal(act, [0, 0, 0.2]): return 5
			if np.array_equal(act, [0, 0, -0.2]): return 6
			
		data = RealData(all_obj=all_obj, path='note_train.txt')
		data = data[:data_size]	
	
	K = 1

	"""
	datas = get_sets(data)
	new_data = get_newdatas(datas)
	data = new_data
	
	print(data[0])
	# sys.exit()
	
	data = [(
		data[i][0][0], 
		# np.concatenate([data[i][1] - data[i-K][1]]),  #, data[i][1] - data[i-K][1]]), 
		# data[i][1] + data[i-K][1], 
		data[i][0][1], 
		discrete_action(data[i][0][2]), 
		# np.concatenate([data[i+1][1] - data[i-K+1][1]]) #data[i+1][1]
		# data[i+1][1],
		data[i][1][1] - data[i][0][1],
	) for i in range(K, len(data)-1)]
	"""
	data = [(
		data[i][0], 
		# np.concatenate([data[i][1] - data[i-K][1]]),  #, data[i][1] - data[i-K][1]]), 
		# data[i][1] + data[i-K][1], 
		data[i][1], 
		discrete_action(data[i][2]), 
		# np.concatenate([data[i+1][1] - data[i-K+1][1]]) #data[i+1][1]
		# data[i+1][1],
		data[i+1][1] #- data[i][1],
	) for i in range(K, len(data)-1)]

	
	for i in range(10): print(data[i])
	
	
	is_valids = []
	for i in range(len(data)):
		is_valids.append(7 - np.count_nonzero(data[i][1] == -1) / 2)
	
	temp_data = []
	for i in range(len(data)):
		if no_zerobeacon:
			if int(is_valids[i]) is not 0:
				temp_data.append((data[i][0], data[i][1], data[i][2], data[i][3], is_valids[i]))
		else:
			temp_data.append((data[i][0], data[i][1], data[i][2], data[i][3], is_valids[i]))
	data = temp_data
	data = data[::skipdata]
	
	
	obs_dim = len(data[0][1])
	act_dim = 1 #len(data[0][2])
	pred_dim = len(data[0][3])
	print(obs_dim, act_dim, pred_dim)
	model = LinearNet()
	
	criterion = torch.nn.CrossEntropyLoss() #MSELoss(reduction='elementwise_mean')
	# optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
	optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
	
	batch_size = 64
	data = [torch.from_numpy(np.array(x)).float() for x in zip(*data)]
	mean_, std_ = data[1].mean(dim=0), data[1].std(dim=0)
	mean3_, std3_ = data[3].mean(dim=0), data[3].std(dim=0)
	data[1] = (data[1] - mean_) / std_
	data[3] = (data[3] - mean3_) / std3_
	dataset = torch.utils.data.TensorDataset(*data)
	
	train_num = int(len(dataset)*0.8)
	valid_num = len(dataset) - train_num
	
	train_dataset = torch.utils.data.TensorDataset(*dataset[:train_num])
	val_dataset = torch.utils.data.TensorDataset(*dataset[train_num:])
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
	
	
	##########################################################
	data_test = [torch.from_numpy(np.array(x)).float() for x in zip(*data_test)]
	data_test[1] = (data_test[1] - mean_) / std_
	data_test[3] = (data_test[3] - mean3_) / std3_
	dataset_test = torch.utils.data.TensorDataset(*data_test)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, drop_last=False, shuffle=False)	
	##########################################################
	
	best_model = None
	best_val_acc = 0.
	for epoch in range(300):
		print('\n ===== Epoch {}\t ====='.format(epoch+1))
		for mode in ['train', 'val']:
			losses = []
			running_corrects = 0
			total_num = 0
			loader = (train_loader if mode == 'train' else val_loader)
			
			result = []
			valid = []
			for i, (dt, obs, act, obs_next, v) in enumerate(loader):
				# print(dt, obs, act, obs_next)
				x1 = obs
				x2 = obs_next
				y = act.long()
				y_pred = model(x1, x2)
				_, preds = torch.max(y_pred.data, 1)
	
				loss = criterion(y_pred, y)
	
				if i == 0:
					print(x1[0].numpy(), x2[0].numpy(), y[0].numpy(), y_pred[0].max(0)[1].numpy())
			
				if mode == 'train':
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				
				losses.append(float(loss))
				running_corrects += torch.sum(preds == y.data)
				total_num += len(y.data)
				
				
				r = preds == y.data
				result += list(r.numpy())
				valid += list(v.numpy())
				
			loss = np.mean(losses)
			acc = running_corrects.numpy() / total_num
			print('{}:\t Loss = {:.4f},\t Acc = {:.4f}'.format(mode, loss, acc))
	
	
			counts = [0, 0, 0, 0, 0, 0, 0, 0]
			sums = [0, 0, 0, 0, 0, 0, 0, 0]
			for i in range(len(result)):
				counts[int(valid[i])] += result[i]
				sums[int(valid[i])] += 1
				
	
			for i in range(8):
				print(sums[i], round(counts[i]/(sums[i]+0.0001), 3))
			print(sum(counts[1:]) / sum(sums[1:]))
			acc = sum(counts[1:]) / sum(sums[1:])
					
			if mode == 'val' and acc > best_val_acc:
				best_val_acc = acc
				best_model = model
				if save_model:
					print('Save model...')
					torch.save(model.state_dict(), model_path)
					torch.save([mean_, std_, mean3_, std3_], model_path_norm)
			print(best_val_acc)
			
	###############################################

		running_corrects1 = 0
		running_sums1 = 0
		running_corrects2 = 0
		running_sums2 = 0

		for i, (dt, obs, act, obs_next, v) in enumerate(test_loader):
			# print(dt, obs, act, obs_next)
			x1 = obs
			x2 = obs_next
			y = act.long()
			y_pred = model(x1, x2)
			_, preds = torch.max(y_pred.data, 1)
		
			loss = criterion(y_pred, y)
		
			if i == 0:
				print(x1[0].numpy(), x2[0].numpy(), y[0].numpy(), y_pred[0].max(0)[1].numpy())
		
		
			
			losses.append(float(loss))
			running_corrects1 += torch.sum(preds == y.data).numpy()
			running_sums1 += len(y.data)

			running_corrects2 += sum((preds == y.data).numpy()*v.numpy())
			running_sums2 += sum(v.numpy())
			
			
			
		loss = np.mean(losses)
		acc1 = running_corrects1 / running_sums1
		acc2 = running_corrects2 / running_sums2

		print('{}:\t Loss = {:.4f},\t Acc1 = {:.4f},\t Acc2 = {:.4f}'.format('Testing', loss, acc1, acc2))

		



###################################################
test_model = None
test_norm = None

def test_initialize():
    global test_model, test_norm
    test_model = LinearNet()
    test_model.load_state_dict(torch.load(model_path))
    test_norm = torch.load(model_path_norm)

def test_query(obs1, obs2):
    global test_model, test_norm
    with torch.no_grad():
        obs1 = torch.from_numpy(obs1).unsqueeze(0)
        obs2 = torch.from_numpy(obs2).unsqueeze(0)
        obsd = obs2 - obs1
        obs1 = (obs1 - test_norm[0]) / test_norm[1]
        obsd = (obsd - test_norm[2]) / test_norm[3]
        
        pred_prob = test_model(obs1, obsd)
        _, pred = pred_prob.max(1)
        pred = int(pred[0].numpy())
    return pred

if __name__ == '__main__':
	train()

	"""

	test_initialize()

	f = open('note_random.txt')
	data = []
	for line in f:
		token = line.strip().split(',')
		data.append((test_action(np.array([float(i) for i in token[1:4]])),np.array([float(i) for i in token[4:]])))
	data = [(data[i][0], data[i][1], data[i+1][1]) for i in range(len(data)-1)]
		
	correct = 0
	for d in data:
		obs1 = d[1].astype('float32')
		obs2 = d[2].astype('float32')
		res = test_query(obs1, obs2)
		correct += (res == d[0])
	print(correct/len(data))
		
	obs1 = np.ones(14, dtype=np.float32)
	obs2 = np.zeros(14, dtype=np.float32)
	"""
