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

t = int(sys.argv[1])


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels.long()] 
	
	
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
			for j in range(i+1+t,min(i+1+t+1,len(new_data))):
				new_datas.append((new_data[i],new_data[j]))
	return new_datas

class LinearNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		hid_dim = 10 #int(sys.argv[1])
		self.linear1 = torch.nn.Linear(14, hid_dim)
		self.linear2 = torch.nn.Linear(hid_dim, hid_dim)
		self.linear3 = torch.nn.Linear(hid_dim, 7)

		
		self.linear4 = torch.nn.Linear(14, hid_dim)
		self.linear5 = torch.nn.Linear(hid_dim, hid_dim)
		self.linear6 = torch.nn.Linear(hid_dim+7, 14)

	def forward_obs(self, obs):
		res = F.relu(self.linear1(obs))
		res = F.relu(self.linear2(res))
		res = self.linear3(res)		
		return res
		
		
		
	def forward_act(self, obs, act):
		res = F.relu(self.linear4(obs))
		res = F.relu(self.linear5(res))
		res = torch.cat([res,act], dim=1)
		res = self.linear6(res)
		return res
		
		
		
	def forward(self, obs1, obs2, act):
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
		pred = self.forward_obs(obs2-obs1)

		
		#query = torch.cat([obs1, one_hot_embedding(act, 7)], dim=1)
		pred_obs = obs1 + self.forward_act(obs1, one_hot_embedding(act, 7))
		
	
		return pred, pred_obs

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
		data[i][1][1] #- data[i][0][1],
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
	"""
	
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
	criterion2 = torch.nn.MSELoss(reduction='elementwise_mean')
	criterion3 = torch.nn.L1Loss()
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
	
	print("="*20)
	print(len(data[0]))
	for epoch in range(300):
		print('\n ===== Epoch {}\t ====='.format(epoch+1))
		for mode in ['train', 'val']:
			losses = []
			losses2 = []
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
				y_pred, obs_pred = model(x1, x2, act)
				_, preds = torch.max(y_pred.data, 1)
	
				loss = criterion(y_pred, y)
				loss2 = criterion2(obs_pred, obs_next)
				
				a = (v != 0).float()
				b = torch.abs((obs_pred-obs > 0).float()-(obs_next-obs > 0).float()).transpose(0,1)
				# b = b[1::2]
				loss3 = torch.sum(torch.mv(b,a)) / torch.sum(torch.mv(torch.ones(b.shape),a)+0.0001)
				# loss3 = torch.sum((v == 0).float() * torch.abs((obs_pred > 0).float()-(obs_next-obs > 0).float())) / (torch.sum((v == 0).float()) * 14)

				#print(obs_pred > 0)
				# print(loss3)
				
				if i == 0:
					print(x1[0].numpy(), x2[0].numpy(), y[0].numpy(), y_pred[0].max(0)[1].numpy())
			
				if mode == 'train':
					optimizer.zero_grad()
					loss2.backward()
					optimizer.step()
				
				losses.append(float(loss2))
				losses2.append(float(loss3))
				running_corrects += torch.sum(preds == y.data)
				total_num += len(y.data)
				
				
				r = preds == y.data
				result += list(r.numpy())
				valid += list(v.numpy())
				
			loss = np.mean(losses)
			loss2 = np.mean(losses2)
			acc = running_corrects.numpy() / total_num
			print('{}:\t Loss = {:.4f},\t Loss2 = {:.4f},\t Acc = {:.4f}'.format(mode, loss, loss2, acc))
	
	
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
	
		for i, (dt, obs, act, obs_next, v) in enumerate(test_loader):
			# print(dt, obs, act, obs_next)
			x1 = obs
			x2 = obs_next
			y = act.long()
			y_pred, obs_pred = model(x1, x2, act)
			_, preds = torch.max(y_pred.data, 1)
	
			loss = criterion(y_pred, y)
			loss2 = criterion2(obs_pred, obs_next)
			
			a = (v != 0).float()
			b = torch.abs((obs_pred-obs > 0).float()-(obs_next-obs > 0).float()).transpose(0,1)
			# b = b[1::2]
			loss3 = torch.sum(torch.mv(b,a)) / torch.sum(torch.mv(torch.ones(b.shape),a)+0.0001)
			# loss3 = torch.sum((v == 0).float() * torch.abs((obs_pred > 0).float()-(obs_next-obs > 0).float())) / (torch.sum((v == 0).float()) * 14)

			#print(obs_pred > 0)
			# print(loss3)
			
			
			losses.append(float(loss2))
			losses2.append(float(loss3))
			running_corrects += torch.sum(preds == y.data)
			total_num += len(y.data)
			
			
			r = preds == y.data
			result += list(r.numpy())
			valid += list(v.numpy())
			
		loss = np.mean(losses)
		loss2 = np.mean(losses2)
		acc = running_corrects.numpy() / total_num
		print('{}:\t Loss = {:.4f},\t Loss2 = {:.4f},\t Acc = {:.4f}'.format('Testing', loss, loss2, acc))
			
			
		"""
		running_corrects1 = 0
		running_sums1 = 0
		running_corrects2 = 0
		running_sums2 = 0
		

		for i, (dt, obs, act, obs_next, v) in enumerate(test_loader):
			# print(dt, obs, act, obs_next)
			x1 = obs
			x2 = obs_next
			y = act.long()
			y_pred, obs_pred = model(x1, x2, act)
			_, preds = torch.max(y_pred.data, 1)
		
			loss = criterion(y_pred, y)
			#loss2 = criterion2(obs_pred, obs_next-obs)
			loss2 = criterion2(obs_pred, obs_next-obs)

			if i == 0:
				print(x1[0].numpy(), x2[0].numpy(), y[0].numpy(), y_pred[0].max(0)[1].numpy())
		
		
			
			losses.append(float(loss2))
			running_corrects1 += torch.sum(preds == y.data).numpy()
			running_sums1 += len(y.data)

			running_corrects2 += sum((preds == y.data).numpy()*v.numpy())
			running_sums2 += sum(v.numpy())
			
			
			
		loss = np.mean(losses)
		acc1 = running_corrects1 / running_sums1
		acc2 = running_corrects2 / running_sums2

		print('{}:\t Loss = {:.4f},\t Acc1 = {:.4f},\t Acc2 = {:.4f}'.format('Testing', loss, acc1, acc2))
		"""
		



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
