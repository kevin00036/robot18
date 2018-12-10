import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData

# data = SimData(10000)
data = RealData()

K = 5
data = [(
    data[i][0], 
    np.concatenate([data[i][1]]),  #, data[i][1] - data[i-K][1]]), 
    # data[i][1] + data[i-K][1], 
    # data[i][1], 
    data[i][2], 
    data[i+1][1]
) for i in range(K, len(data)-1)]

for i in range(10): print(data[i])


class LinearNet(torch.nn.Module):
	def __init__(self, D_in, D_out):
		super().__init__()
		self.linear = torch.nn.Linear(D_in, 15)
		self.linear2 = torch.nn.Linear(15, 15)
		self.linear3 = torch.nn.Linear(15, D_out)
	
	def forward(self, x):
		x = self.linear(x)
		x = F.relu(x)
		#x = self.linear2(x)
		#x = F.relu(x)
		x = self.linear3(x)
		return x


obs_dim = len(data[0][1])
act_dim = len(data[0][2])
pred_dim = len(data[0][3])
print(obs_dim, act_dim, pred_dim)
model = LinearNet(obs_dim + obs_dim, act_dim) #, pred_dim)

criterion = torch.nn.MSELoss(reduction='elementwise_mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

batch_size = 32
data = [torch.from_numpy(np.array(x)).float() for x in zip(*data)]
mean_, std_ = data[1].mean(dim=0), data[1].std(dim=0)
mean3_, std3_ = data[3].mean(dim=0), data[3].std(dim=0)
data[1] = (data[1] - mean_) / std_
data[3] = (data[3] - mean3_) / std3_
dataset = torch.utils.data.TensorDataset(*data)

train_num = int(len(dataset)*0.8)

train_dataset = torch.utils.data.TensorDataset(*dataset[:train_num])
val_dataset = torch.utils.data.TensorDataset(*dataset[train_num:])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False)

for epoch in range(30000):
	train_losses = []
	val_losses = []
	
	for i, (dt, obs, act, obs_next) in enumerate(train_loader):
		# print(dt, obs, act, obs_next)
		# Forward pass: Compute predicted y by passing x to the model

		x = torch.cat([obs, obs_next], dim=1)
		y = act
		y_pred = model(x)
		
		# Compute and print loss
		loss = criterion(y_pred, y)
		# print(epoch, loss.item())
	
		if i == 0:
			# print(float(loss))
			# print((y - y_pred).pow(2).sum())
			print(x[0], y[0], y_pred[0].data)
	
		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	
		# train_losses.append(float(criterion(y_pred * std_, y * std_)))
		train_losses.append(float(loss))
	
	with torch.no_grad():
		for i, (dt, obs, act, obs_next) in enumerate(val_loader):
			# print(dt, obs, act, obs_next)
			# Forward pass: Compute predicted y by passing x to the model
	
			x = torch.cat([obs, obs_next], dim=1)
			y = act
			y_pred = model(x)
	
			# if i == 0 and epoch == 19:
				# for z in range(batch_size):
					# print(x[z][:-3]*std_+mean_, x[z][-3:], y[z]*std3_+mean3_, y_pred[z].data*std3_+mean3_)
	
			# Compute and print loss
			loss = criterion(y_pred, y)
			val_losses.append(float(loss))
			# val_losses.append(float(criterion(y_pred * std_, y * std_)))
	
	train_loss = np.mean(train_losses)
	val_loss = np.mean(val_losses)
	print(f'{epoch:3d}: Train={train_loss:.5f}, Val={val_loss:.5f}')
    
