import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import random
from data import SimData, RealData

dev = 'cuda'

data = SimData(10000)
# data = RealData()

class SequenceDataset(Dataset):
    def __init__(self, data):
        self.seq_len = 10
        self.tot_len = len(data[0]) - self.seq_len + 1
        self.data = data
        
    def __len__(self):
        return self.tot_len

    def __getitem__(self, idx):
        return tuple(x[idx:idx+self.seq_len] for x in self.data)


class NormalRNN(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.hidden_dim = 10
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gru = nn.GRU(obs_dim + act_dim, self.hidden_dim, batch_first=True)

        self.linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.obs_dim)
        # self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x, dt):
        x, _ = self.gru(x)
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        return x
    
class OtherRNN(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.hidden_dim = 10
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.guess_hidden = torch.nn.Linear(self.obs_dim, self.hidden_dim)

        self.recur_linear = torch.nn.Linear(obs_dim + act_dim + self.hidden_dim, self.hidden_dim)

        self.linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear2 = torch.nn.Linear(self.hidden_dim, self.obs_dim)
        # self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x, dt):
        l = x.shape[1]
        # h = torch.zeros((x.shape[0], self.hidden_dim), dtype=torch.float32, device=dev)
        h = self.guess_hidden(x[:, 0, :-3])
        out = []
        for i in range(l):
            inp = torch.cat([x[:, i], h], dim=-1)
            # print(inp.shape, dt.shape)
            dh = dt[:, i].unsqueeze(-1) * self.recur_linear(inp)
            h = h + dh
            o = self.linear2(F.relu(self.linear(h)))
            out.append(o)

        out = torch.stack(out, dim=1)
        return out


obs_dim = len(data[0][1])
act_dim = len(data[0][2])
print(obs_dim, act_dim)
# model = NormalRNN(obs_dim, act_dim).to(dev)
model = OtherRNN(obs_dim, act_dim).to(dev)

criterion = torch.nn.MSELoss(reduction='elementwise_mean')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

batch_size = 32
train_num = int(len(data)*0.8)
print(train_num)

data = [torch.from_numpy(np.array(x)).float().to(dev) for x in zip(*data)]
mean_, std_ = data[1].mean(dim=0), data[1].std(dim=0)
data[1] = (data[1] - mean_) / std_


train_dataset = SequenceDataset([x[:train_num] for x in data])
val_dataset = SequenceDataset([x[train_num:] for x in data])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

for epoch in range(50):
    train_losses = []
    val_losses = []
    
    for i, (dt, obs, act) in enumerate(train_loader):

        x = torch.cat([obs, act], dim=-1)[:,:-1]
        y = obs[:,1:]
        dt = dt[:,:-1]
        y_pred = model(x, dt)

        # print(y_pred.shape, y.shape)

        # Compute and print loss
        loss = criterion(y_pred, y)

        # if i == 0:
            # print(x[0], y[0], y_pred[0].data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_losses.append(float(criterion(y_pred * std_, y * std_)))
        train_losses.append(float(loss))

    with torch.no_grad():
        for i, (dt, obs, act) in enumerate(val_loader):

            x = torch.cat([obs, act], dim=-1)[:,:-1]
            y = obs[:,1:]
            dt = dt[:,:-1]
            y_pred = model(x, dt)

            # Compute and print loss
            loss = criterion(y_pred, y)

            if i == 0:
                print(x[0, -5:])
                print(y[0, -5:])
                print(y_pred[0, -5:].data)

            val_losses.append(float(loss))

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    print(f'{epoch:3d}: Train={train_loss:.5f}, Val={val_loss:.5f}')
    
