import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import random
from data import SimData, RealData
from render import *

dev = 'cpu'

data = SimData(20000, all_obj=True)
# data = RealData(all_obj=True)

obs_dim = len(data[0][1])
act_dim = len(data[0][2])
evald_ = torch.from_numpy(np.array([x[1] for x in data], dtype=np.float32)) # obs
eval_mean_ = evald_.mean(dim=0)
eval_std_ = evald_.std(dim=0)

train_num = int(len(data)*0.8)
train_data = data[:train_num]
val_data = data[train_num:]

def preproc_sequence(d, seq_len=10):
    ret = []
    num = len(d[0])
    for i in range(len(d) - seq_len + 1):
        lst = []
        for j in range(num):
            r = [x[j] for x in d[i:i+seq_len]]
            lst.append(r)
        ret.append(tuple(lst))
    return ret

datas = {
    'train': preproc_sequence(train_data),
    'val': preproc_sequence(val_data),
    'val_n': preproc_sequence(val_data, seq_len=2),
}

for k in datas:
    datas[k]= [torch.from_numpy(np.array(x)).float().to(dev) for x in zip(*datas[k])]

normalize_idxs = [1, 2] # Which indexes (ex: dt, obs, act) to normalize
means_, stds_ = {}, {}
ddim = 2 # How many prefix dims to normalize along, usually 1. (2 when the data are sequences)
for i in normalize_idxs:
    means_[i] = datas['train'][i].reshape(-1, *datas['train'][i].shape[ddim:]).mean(dim=0)
    stds_[i] = datas['train'][i].reshape(-1, *datas['train'][i].shape[ddim:]).std(dim=0)

def normalize(d, inv=False):
    r = []
    for i in range(len(d)):
        if i in normalize_idxs:
            if not inv:
                r.append((d[i] - means_[i]) / stds_[i])
            else:
                r.append((d[i] * stds_[i]) + means_[i])
        else:
            r.append(d[i])
    return r

for k in datas:
    datas[k] = normalize(datas[k])

batch_size = 32
datasets = {}
loaders = {}
for k in datas:
    datasets[k] = torch.utils.data.TensorDataset(*datas[k])
    loaders[k] = torch.utils.data.DataLoader(
        datasets[k],
        batch_size=batch_size,
        drop_last=True,
        shuffle=(k == 'train'),
    )

print('=== Data processing completed ===')


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

    def forward(self, x, dt, output_hidden=False, input_hidden=None):
        l = x.shape[1]
        # h = torch.zeros((x.shape[0], self.hidden_dim), dtype=torch.float32, device=dev)
        h = self.guess_hidden(x[:, 0, :-3])
        if input_hidden is not None:
            h = input_hidden
        out = []
        for i in range(l):
            inp = torch.cat([x[:, i], h], dim=-1)
            # print(inp.shape, dt.shape)
            # dh = dt[:, i].unsqueeze(-1) * self.recur_linear(inp)
            # h = h + dh
            h = self.recur_linear(inp)
            o = self.linear2(F.relu(self.linear(h)))
            out.append(o)

        out = torch.stack(out, dim=1)
        if output_hidden:
            return out, h
        else:
            return out

model = NormalRNN(obs_dim, act_dim).to(dev)
# model = OtherRNN(obs_dim, act_dim).to(dev)

criterion = torch.nn.MSELoss(reduction='elementwise_mean')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate_obs_dis(y_pred, y):
    ypn = (y_pred - eval_mean_) / eval_std_
    yn = (y - eval_mean_) / eval_std_
    dis = ((ypn - yn) ** 2).mean()
    return dis.data.numpy()

for epoch in range(10):
    print('\n ===== Epoch {}\t ====='.format(epoch+1))
    for mode in datas:
        losses = []
        eval_diss = []
        loader = loaders[mode]
        
        for i, (dt, obs, act) in enumerate(loader):
            # x = torch.cat([obs, act], dim=-1)
            # y = obs[:,1:]
            # y_pred = obs + model(x, dt)

            y = obs[:,1:]
            last_obs = obs[:, 0:1]
            y_pred = []
            for t in range(obs.shape[1]):
                x = torch.cat([last_obs, act[:, t:t+1]], dim=-1)
                d = dt[:, t:t+1]
                yp = last_obs + model(x, dt)
                y_pred.append(yp)
                last_obs = yp
            y_pred = torch.cat(y_pred, dim=1)

            y_pred = y_pred[:, :-1]

            # Compute and print loss
            loss = criterion(y_pred, y)

            y_pred_unnorm = (y_pred * stds_[1]) + means_[1]
            y_unnorm = (y * stds_[1]) + means_[1]
            eval_dis = evaluate_obs_dis(y_pred_unnorm, y_unnorm)
            eval_diss.append(eval_dis)

            # if i == 0:
                # print(x[0], y[0], y_pred[0].data)

            # Zero gradients, perform a backward pass, and update the weights.
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(float(loss))

        loss = np.mean(losses)
        eval_dis = np.mean(eval_diss)
        print('{}:\t Loss = {:.4f}\t Dis = {:.4f}'.format(mode, loss, eval_dis))

# h = torch.zeros((1, model.hidden_dim), dtype=torch.float32, device=dev)
for i, (dt, obs, act) in enumerate(datasets['val_n']):
    x = torch.cat([obs, act], dim=-1).unsqueeze(0)
    dt = dt.unsqueeze(0)
    # y_pred, nh = model(x, dt, output_hidden=True, input_hidden=h)
    y_pred = model(x, dt)
    y_pred = (obs + y_pred).data.squeeze(0)
    # h = nh

    y_pred = (y_pred[0] * stds_[1]) + means_[1]
    obs = (obs[0] * stds_[1]) + means_[1]

    reset_frame()
    render_objs(obs)
    render_objs(y_pred, rad=5)
    render_show()


