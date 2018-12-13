import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData
from render import *

dev = 'cpu'

# data = SimData(20000, all_obj=True)
data = RealData(all_obj=True)

evald_ = torch.from_numpy(np.array([x[1] for x in data], dtype=np.float32)) # obs
eval_mean_ = evald_.mean(dim=0)
eval_std_ = evald_.std(dim=0)

train_num = int(len(data)*0.8)
train_data = data[:train_num]
val_data = data[train_num:]

def preproc(d):
    def calc_seen(x):
        return np.float32(x[::2] != -1.0)
    K = 1
    data = [(
        d[i][0], 
        d[i][1],
        # np.concatenate([d[i][1], d[i][1] - d[i-K][1]]), 
        # data[i][1] + data[i-K][1], 
        d[i][1], 
        d[i][2], 
        d[i+1][1],
        calc_seen(d[i][1]),
        calc_seen(d[i+1][1]),
    ) for i in range(K, len(d)-1)]
    return data

datas = {
    'train': preproc(train_data),
    'val': preproc(val_data),
}

for k in datas:
    datas[k]= [torch.from_numpy(np.array(x)).float().to(dev) for x in zip(*datas[k])]

normalize_idxs = [1, 2, 3, 4] # Which indexes (ex: dt, obs, act) to normalize
means_, stds_ = {}, {}
ddim = 1 # How many prefix dims to normalize along, usually 1. (2 when the data are sequences)
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

class LinearNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        x = self.linear(x)
        return x

class LinearNetShared(torch.nn.Module):
    def __init__(self, D_obs, D_act, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2)

    def forward(self, obs, act):
        bs = obs.shape[0]
        obs = obs.reshape(bs, -1, 2)
        act = act.unsqueeze(1).expand(-1, obs.shape[1], -1)
        inp = torch.cat([obs, act], dim=-1)
        res = self.linear(inp)
        res = res.reshape(bs, -1)
        return res

class FeedForwardNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(D_in, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, D_out)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

class FeedForwardNetShared(torch.nn.Module):
    def __init__(self, D_obs, D_act, D_out):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 2)

    def forward(self, obs, act):
        bs = obs.shape[0]
        obs = obs.reshape(bs, -1, 2)
        act = act.unsqueeze(1).expand(-1, obs.shape[1], -1)
        inp = torch.cat([obs, act], dim=-1)
        res = self.linear(inp)
        res = F.relu(res)
        res = self.linear2(res)
        res = F.relu(res)
        res = self.linear3(res)
        res = res.reshape(bs, -1)
        return res

class FeedForwardNetClass(torch.nn.Module):
    def __init__(self, D_obs, D_act, D_out):
        super().__init__()
        hid_dim = 10
        self.linear_fo = torch.nn.Linear(2, hid_dim)
        self.linear_fs = torch.nn.Linear(1, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim + D_act, hid_dim)
        self.linear_o = torch.nn.Linear(hid_dim, 2)
        self.linear_s = torch.nn.Linear(hid_dim, 1)

    def forward(self, obs, act, seen):
        bs = obs.shape[0]
        obs = obs.reshape(bs, -1, 2)
        act = act.unsqueeze(1).expand(-1, obs.shape[1], -1)
        seen = seen.unsqueeze(2)

        fo = self.linear_fo(obs)
        fs = self.linear_fs(seen * 0.)
        f = seen * fo + (1-seen) * fs

        inp = torch.cat([f, act], dim=-1)
        res = self.linear2(inp)
        res = F.relu(res)
        o = self.linear_o(res)
        s = torch.sigmoid(self.linear_s(res))

        o = o.reshape(bs, -1)
        s = s.reshape(bs, -1)
        return o, s

obs_dim = datas['train'][2].shape[-1]
act_dim = datas['train'][3].shape[-1]
pred_dim = datas['train'][4].shape[-1]
print(obs_dim, act_dim, pred_dim)
# model = LinearNet(obs_dim + act_dim, pred_dim)
# model = LinearNetShared(obs_dim, act_dim, pred_dim)
# model = FeedForwardNet(obs_dim + act_dim, pred_dim)
# model = FeedForwardNetShared(obs_dim, act_dim, pred_dim)
model = FeedForwardNetClass(obs_dim, act_dim, pred_dim)

criterion = torch.nn.MSELoss(reduction='elementwise_mean')
mse_loss_unreduced = torch.nn.MSELoss(reduction='none')
bce_loss = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate_obs_dis(y_pred, y):
    ypn = (y_pred - eval_mean_) / eval_std_
    yn = (y - eval_mean_) / eval_std_
    dis = ((ypn - yn) ** 2).mean()
    return dis.data.numpy()


for epoch in range(20):
    print('\n ===== Epoch {}\t ====='.format(epoch+1))
    for mode in datas:
        losses = []
        eval_diss = []
        loader = loaders[mode]

        for i, (dt, obs, obs_aug, act, obs_next, seen, seen_next) in enumerate(loader):
            # y = obs_next
            # y_pred = obs + model(obs_aug, act)

            y = obs_next
            dobs, seen_pred = model(obs_aug, act, seen)
            y_pred = obs + dobs
            mask = (seen_pred > 0.5).unsqueeze(2).expand(-1, -1, 2).reshape(obs.shape[0], -1).float()
            minus_one_normed = (-1.0 - means_[1]) / stds_[1]
            y_pred = mask * y_pred + (1-mask) * minus_one_normed

            # Compute and print loss
            # loss = criterion(y_pred, y)
            lamda = 1.
            loss = (lamda * bce_loss(seen_pred, seen_next) + 
                    (seen_next * mse_loss_unreduced(y_pred, y).reshape(obs.shape[0], -1, 2).mean(dim=2)).mean())

            y_pred_unnorm = (y_pred * stds_[1]) + means_[1]
            y_unnorm = (y * stds_[1]) + means_[1]
            eval_dis = evaluate_obs_dis(y_pred_unnorm, y_unnorm)
            eval_diss.append(float(eval_dis))

            # Zero gradients, perform a backward pass, and update the weights.
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(float(loss))

        loss = np.mean(losses)
        eval_dis = np.mean(eval_diss)
        print('{}:\t Loss = {:.4f}\t Dis = {:.4f}'.format(mode, loss, eval_dis))

for i, (dt, obs, obs_aug, act, obs_next, seen, seen_next) in enumerate(datasets['val']):
    if i >= 1000: break

    # y_pred = obs + model(obs_aug.unsqueeze(0), act.unsqueeze(0)).squeeze(0) * 5

    dobs, seen_pred = model(obs_aug.unsqueeze(0), act.unsqueeze(0), seen.unsqueeze(0))
    y_pred = obs + dobs.squeeze(0) * 3
    mask = (seen_pred > 0.5).unsqueeze(2).expand(-1, -1, 2).reshape(-1).float()
    minus_one_normed = (-1.0 - means_[1]) / stds_[1]
    y_pred = mask * y_pred + (1-mask) * minus_one_normed

    y_pred = y_pred.data
    y_pred = (y_pred * stds_[1]) + means_[1]
    obs = (obs * stds_[1]) + means_[1]
    act = (act * stds_[3]) + means_[3]

    reset_frame()
    render_objs(obs)
    render_objs(y_pred, rad=5, outline=True)
    render_act(discrete_action(act))
    render_show()
    # img_path = 'imgs/forward_vis/'
    # img_name = img_path + '{:03d}.jpg'.format(i)
    # render_save(img_name)


