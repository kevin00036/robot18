import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData

# data = SimData(10000)
data = RealData()

cnt = 0
for d in data:
    if d[1][0] == -1:
        cnt += 1
print(cnt, '/', len(data))

K = 1
data = [(
    data[i][0], 
    np.concatenate([data[i][1], data[i][1] - data[i-K][1]]), 
    # data[i][1] + data[i-K][1], 
    # data[i][1], 
    data[i][2], 
    data[i+1][1] - data[i][1],
) for i in range(K, len(data)-1)]

for i in range(20): print(data[i])
# exit()

class LinearNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        hidden_dim = 30
        self.linear = torch.nn.Linear(D_in, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, D_out)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


obs_dim = len(data[0][1])
act_dim = len(data[0][2])
pred_dim = len(data[0][3])
print(obs_dim, act_dim)
model = LinearNet(obs_dim + pred_dim, act_dim)

criterion = torch.nn.SmoothL1Loss(reduction='elementwise_mean')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

batch_size = 32
data = [torch.from_numpy(np.array(x)).float() for x in zip(*data)]
mean_, std_ = data[1].mean(dim=0), data[1].std(dim=0)
mean3_, std3_ = data[3].mean(dim=0), data[3].std(dim=0)
data[1] = (data[1] - mean_) / std_
data[3] = (data[3] - mean3_) / std3_
dataset = torch.utils.data.TensorDataset(*data)

# for i in range(10): print(data[i])

train_num = int(len(dataset)*0.8)

train_dataset = torch.utils.data.TensorDataset(*dataset[:train_num])
val_dataset = torch.utils.data.TensorDataset(*dataset[train_num:])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False)

for epoch in range(200):
    train_losses = []
    val_losses = []
    
    for mode in ['train', 'val']:
        loader = (train_loader if mode == 'train' else val_loader)
        for i, (dt, obs, act, obs_next) in enumerate(loader):
            # print(dt, obs, act, obs_next)

            x = torch.cat([obs, obs_next], dim=1)
            y = act
            y_pred = model(x)
            # print(x.shape, y.shape, y_pred.shape)

            loss = criterion(y_pred, y)

            if i == 0:
                print(x[0], y[0], y_pred[0].data)

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(float(loss))
            else:
                val_losses.append(float(loss))


    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    print(f'{epoch:3d}: Train={train_loss:.5f}, Val={val_loss:.5f}')
    
