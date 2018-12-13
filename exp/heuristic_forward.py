import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData
from render import *

dev = 'cpu'

data = SimData(20000, all_obj=True)
# data = RealData(all_obj=True)

evald_ = torch.from_numpy(np.array([x[1] for x in data], dtype=np.float32)) # obs
eval_mean_ = evald_.mean(dim=0)
eval_std_ = evald_.std(dim=0)

train_num = int(len(data)*0.8)
train_data = data[:train_num]
val_data = data[train_num:]

def preproc(d):
    def calc_seen(x):
        return np.float32(x[::2] != -1.0)
    data = [(
        d[i][0], 
        d[i][1],
        d[i][2], 
        d[i+1][1],
        calc_seen(d[i][1]),
        calc_seen(d[i+1][1]),
    ) for i in range(len(d)-1)]
    return data

datas = {
    'train': preproc(train_data),
    'val': preproc(val_data),
}

for k in datas:
    datas[k]= [torch.from_numpy(np.array(x)).float().to(dev) for x in zip(*datas[k])]

normalize_idxs = [] # Which indexes (ex: dt, obs, act) to normalize
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
        self.moves = torch.nn.Parameter(torch.zeros((1, 1, 6, 2)))
        self.angle_thres = torch.nn.Parameter(torch.ones(()) * np.pi * 0.25)

    def forward(self, obs, act, seen, mult=1.0):
        bs = obs.shape[0]
        obs = obs.reshape(bs, -1, 2)
        # x = self.linear(x)

        res = torch.zeros_like(obs)
        mo = -torch.ones_like(obs)

        cact = act.unsqueeze(1).expand(-1, obs.shape[1], -1)

        xx = obs[:,:,0] * torch.cos(obs[:,:,1])
        yy = obs[:,:,0] * torch.sin(obs[:,:,1])
        xy = torch.stack([xx, yy], dim=2)

        def xy_to_db(obs_):
            dd = obs_.norm(2, dim=2)
            bb = torch.atan2(obs_[:,:,1], obs_[:,:,0])
            db = torch.stack([dd, bb], dim=2)
            return db

        # moves = [(-10, 0), (10, 0), (0, -10), (0, 10), (0, -0.1), (0, 0.1)]
        # moves = torch.from_numpy(np.array(moves)).float().unsqueeze(0).unsqueeze(0)
        moves = self.moves * mult
        angle_thres = self.angle_thres

        # 0
        res = torch.where(cact.sum(2, keepdim=True) == 0, obs, res)

        # 1
        nobs = xy_to_db(xy + moves[:,:,0,:])
        res = torch.where(cact[:,:,0:1] > 0, nobs, res)
        # 2
        nobs = xy_to_db(xy + moves[:,:,1,:])
        res = torch.where(cact[:,:,0:1] < 0, nobs, res)
        # 3
        nobs = xy_to_db(xy + moves[:,:,2,:])
        res = torch.where(cact[:,:,1:2] > 0, nobs, res)
        # 4
        nobs = xy_to_db(xy + moves[:,:,3,:])
        res = torch.where(cact[:,:,1:2] < 0, nobs, res)
        # 5
        nobs = obs + moves[:,:,4,:]
        res = torch.where(cact[:,:,2:3] > 0, nobs, res)
        # 6
        nobs = obs + moves[:,:,5,:]
        res = torch.where(cact[:,:,2:3] < 0, nobs, res)

        # -1 mask
        res = torch.where(obs[:,:,0:1] == -1, obs, res)
        
        # angle mask
        res = torch.where(res[:,:,1:2].abs() > angle_thres, mo, res)

        res = res.reshape(bs, -1)
        return res

obs_dim = datas['train'][2].shape[-1]
act_dim = datas['train'][3].shape[-1]
pred_dim = datas['train'][4].shape[-1]
print(obs_dim, act_dim, pred_dim)
model = LinearNet(obs_dim + act_dim, pred_dim)

criterion = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

        for i, (dt, obs, act, obs_next, seen, seen_next) in enumerate(loader):
            y = obs_next
            y_pred = model(obs, act, seen)

            # Compute and print loss
            mask = (seen * seen_next).float().unsqueeze(2).expand(-1, -1, 2).reshape(obs.shape[0], -1)
            loss = (criterion(y_pred, y) * mask).mean()

            # y_pred_unnorm = (y_pred * stds_[1]) + means_[1]
            # y_unnorm = (y * stds_[1]) + means_[1]
            y_pred_unnorm = y_pred
            y_unnorm = y
            eval_dis = evaluate_obs_dis(y_pred_unnorm, y_unnorm)
            eval_diss.append(float(eval_dis))

            # Zero gradients, perform a backward pass, and update the weights.
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(float(loss))

        if mode == 'train':
            print(model.moves.squeeze(0).squeeze(0).data)
        loss = np.mean(losses)
        eval_dis = np.mean(eval_diss)
        print('{}:\t Loss = {:.4f}\t Dis = {:.4f}'.format(mode, loss, eval_dis))

for i, (dt, obs, act, obs_next, seen, seen_next) in enumerate(datasets['val']):
    # x = torch.cat([obs_aug, act], dim=-1).unsqueeze(0)
    y_pred = model(obs.unsqueeze(0), act.unsqueeze(0), seen.unsqueeze(0), mult=3.0).squeeze(0)
    y_pred = y_pred.data

    # y_pred = (y_pred * stds_[1]) + means_[1]
    # obs = (obs * stds_[1]) + means_[1]
    # act = (act * stds_[3]) + means_[3]

    reset_frame()
    render_objs(obs)
    render_objs(y_pred, rad=5, outline=True)
    render_act(discrete_action(act))
    # render_show(100)
    img_path = 'imgs/forward_heu/'
    img_name = img_path + '{:03d}.jpg'.format(i)
    render_save(img_name)


