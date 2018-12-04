import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData
import sys

model_path = 'models/inverse_classify.mdl'
model_path_norm = model_path + '.norm'

class LinearNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hid_dim = 10
        self.linear = torch.nn.Linear(4, hid_dim)
        self.linear2 = torch.nn.Linear(hid_dim, hid_dim)
        self.linear3 = torch.nn.Linear(hid_dim, 7)

    def forward(self, obs1, obs2):
        obs1 = [r for r in torch.split(obs1, 2, dim=1)]
        obs1 = torch.stack(obs1, dim=1)

        obs2 = [r for r in torch.split(obs2, 2, dim=1)]
        obs2 = torch.stack(obs2, dim=1)

        obs = torch.cat([obs1, obs2], dim=2) #(32, 6, 4)

        res = F.relu(self.linear(obs))
        res = F.relu(self.linear2(res))

        pred = self.linear3(res) #(32, 6, 7)
        pred = pred.sum(dim=1)

        return pred

def train():
    source = 'sim'
    # source = 'real'
    data_size = 10000
    all_obj = True
    # all_obj = False

    save_model = False
    # save_model = True

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
            if np.array_equal(act, [0.3, 0, 0]): return 1
            if np.array_equal(act, [-0.3, 0, 0]): return 2
            if np.array_equal(act, [0, 0.3, 0]): return 3
            if np.array_equal(act, [0, -0.3, 0]): return 4
            if np.array_equal(act, [0, 0, 0.3]): return 5
            if np.array_equal(act, [0, 0, -0.3]): return 6
            
        data = RealData(all_obj=all_obj)


    K = 1
    data = [(
        data[i][0], 
        # np.concatenate([data[i][1] - data[i-K][1]]),  #, data[i][1] - data[i-K][1]]), 
        # data[i][1] + data[i-K][1], 
        data[i][1], 
        discrete_action(data[i][2]), 
        # np.concatenate([data[i+1][1] - data[i-K+1][1]]) #data[i+1][1]
        # data[i+1][1],
        data[i+1][1] - data[i][1],
    ) for i in range(K, len(data)-1)]

    for i in range(10): print(data[i])




    obs_dim = len(data[0][1])
    act_dim = 1 #len(data[0][2])
    pred_dim = len(data[0][3])
    print(obs_dim, act_dim, pred_dim)
    model = LinearNet()

    criterion = torch.nn.CrossEntropyLoss() #MSELoss(reduction='elementwise_mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)

    batch_size = 32
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
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, drop_last=False, shuffle=True)


    best_val_acc = 0.
    for epoch in range(30000):
        print('\n ===== Epoch {}\t ====='.format(epoch+1))
        for mode in ['train', 'val']:
            losses = []
            running_corrects = 0
            total_num = 0
            loader = (train_loader if mode == 'train' else val_loader)
            for i, (dt, obs, act, obs_next) in enumerate(loader):
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

            loss = np.mean(losses)
            acc = running_corrects.numpy() / total_num
            print('{}:\t Loss = {:.4f},\t Acc = {:.4f}'.format(mode, loss, acc))

            if mode == 'val' and acc > best_val_acc:
                best_val_acc = acc
                if save_model:
                    print('Save model...')
                    torch.save(model.state_dict(), model_path)
                    torch.save([mean_, std_, mean3_, std3_], model_path_norm)

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
    # train()
    test_initialize()
    obs1 = np.zeros(14, dtype=np.float32)
    obs2 = np.ones(14, dtype=np.float32)
    res = test_query(obs1, obs2)
    print(res)
