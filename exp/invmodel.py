import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data_many import SimData, RealData
import sys


source = 'sim'

if source == 'sim':
	def discrete_action(act):
		if np.array_equal(act, [0, 0, 0]): return 0
		if np.array_equal(act, [0.5, 0, 0]): return 1
		if np.array_equal(act, [-0.5, 0, 0]): return 2
		if np.array_equal(act, [0, 0.5, 0]): return 3
		if np.array_equal(act, [0, -0.5, 0]): return 4
		if np.array_equal(act, [0, 0, 0.5]): return 5
		if np.array_equal(act, [0, 0, -0.5]): return 6

	data = SimData(10000)
	
else:

	def discrete_action(act):
		if np.array_equal(act, [0, 0, 0]): return 0
		if np.array_equal(act, [0.3, 0, 0]): return 1
		if np.array_equal(act, [-0.3, 0, 0]): return 2
		if np.array_equal(act, [0, 0.3, 0]): return 3
		if np.array_equal(act, [0, -0.3, 0]): return 4
		if np.array_equal(act, [0, 0, 0.3]): return 5
		if np.array_equal(act, [0, 0, -0.3]): return 6
		
	data = RealData()


	
#data = SimData(10000)
data = RealData()

K = 1  #  ]), #
data = [(
    data[i][0], 
    np.concatenate([data[i][1] - data[i-K][1]]),  #, data[i][1] - data[i-K][1]]), 
    # data[i][1] + data[i-K][1], 
    # data[i][1], 
    discrete_action(data[i][2]), 
    np.concatenate([data[i+1][1] - data[i-K+1][1]]) #data[i+1][1]
) for i in range(K, len(data)-1)]

for i in range(10): print(data[i])


N = int(sys.argv[1])

class LinearNet(torch.nn.Module):
	def __init__(self, D_in1, D_in2, D_out):
		super().__init__()
		self.linear = torch.nn.Linear(D_in1, N)
		self.linear2 = torch.nn.Linear(N, N)
		self.linear3 = torch.nn.Linear(2*N, 7)
	
		#self.linear3 = torch.nn.Linear(2*D_in1, 7)
	def forward(self, x1, x2):
		
		x1 = self.linear(x1)
		x2 = self.linear(x2)

		x1 = F.relu(x1)
		x2 = F.relu(x2)

		x1 = self.linear2(x1)
		x2 = self.linear2(x2)

		x1 = F.relu(x1)
		x2 = F.relu(x2)

		x = torch.cat([x1, x2], dim=1)
		x = self.linear3(x)
		return x


obs_dim = len(data[0][1])
act_dim = 1 #len(data[0][2])
pred_dim = len(data[0][3])
print(obs_dim, act_dim, pred_dim)
model = LinearNet(obs_dim, obs_dim, act_dim) #, pred_dim)

criterion = torch.nn.CrossEntropyLoss() #MSELoss(reduction='elementwise_mean')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1) #lr=1e-4)

batch_size = 32
data = [torch.from_numpy(np.array(x)).float() for x in zip(*data)]
mean_, std_ = data[1].mean(dim=0), data[1].std(dim=0)
mean3_, std3_ = data[3].mean(dim=0), data[3].std(dim=0)
data[1] = (data[1] - mean_) / std_
data[3] = (data[3] - mean3_) / std3_
dataset = torch.utils.data.TensorDataset(*data)

train_num = int(len(dataset)*0.8)
valid_num = int(len(dataset)) - train_num

train_dataset = torch.utils.data.TensorDataset(*dataset[:train_num])
val_dataset = torch.utils.data.TensorDataset(*dataset[train_num:])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False)

best_val = 0
for epoch in range(30000):
	train_losses = []
	val_losses = []
	
	running_corrects = 0
	for i, (dt, obs, act, obs_next) in enumerate(train_loader):
		# print(dt, obs, act, obs_next)
		# Forward pass: Compute predicted y by passing x to the model
	
		#x = torch.cat([obs, obs_next], dim=1)
		x1 = obs #torch.cat([obs, obs_next], dim=1)
		x2 = obs_next
		y = act.long()
		y_pred = model(x1, x2)
		_, preds = torch.max(y_pred.data, 1)
		# Compute and print loss
		
		loss = criterion(y_pred, y)
		# print(epoch, loss.item())
	
		if i == 0:
			# print(float(loss))
			# print((y - y_pred).pow(2).sum())
			print(x1[0], x2[0], y[0], y_pred[0].data)
	
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
		# train_losses.append(float(criterion(y_pred * std_, y * std_)))
		train_losses.append(float(loss))
		running_corrects += torch.sum(preds == y.data)
	print(running_corrects.numpy()/train_num)
		
	with torch.no_grad():
		running_corrects = 0
		for i, (dt, obs, act, obs_next) in enumerate(val_loader):
			# print(dt, obs, act, obs_next)
			# Forward pass: Compute predicted y by passing x to the model
	
			x1 = obs #torch.cat([obs, obs_next], dim=1)
			x2 = obs_next
			y = act.long()
			y_pred = model(x1, x2)


			#y_pred = torch.rand(y_pred.shape)
			_, preds = torch.max(y_pred.data, 1)

			
			#if i == 0:
				# print(float(loss))
				# print((y - y_pred).pow(2).sum())
				# print(x1[0], x2[0], y[0], y_pred[0].data)
				
			"""
			print(y)
			print(preds)
			print('\n\n')
			"""
			
			
			# if i == 0 and epoch == 19:
				# for z in range(batch_size):
					# print(x[z][:-3]*std_+mean_, x[z][-3:], y[z]*std3_+mean3_, y_pred[z].data*std3_+mean3_)
	
			# Compute and print loss
			loss = criterion(y_pred, y)
			val_losses.append(float(loss))
			# val_losses.append(float(criterion(y_pred * std_, y * std_)))
			running_corrects += torch.sum(preds == y.data)
		print(running_corrects.numpy()/valid_num)
		val_acc = running_corrects.numpy()/valid_num
	
	train_loss = np.mean(train_losses)
	val_loss = np.mean(val_losses)
	if val_acc > best_val:
		best_val = val_acc
	print(best_val)
	print(f'{epoch:3d}: Train={train_loss:.5f}, Val={val_loss:.5f}')
    
