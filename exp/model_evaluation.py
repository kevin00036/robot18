import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from data import SimData, RealData
import sys

# model_path = 'models/inverse_classify_sim_consistency0.1.mdl'
model_path = 'models/inverse_classify_sim_aug20.mdl'
model_path_norm = model_path + '.norm'

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels.long()] 

class LinearNet(torch.nn.Module):
	def __init__(self):
		super().__init__()
		hid_dim = 10 #int(sys.argv[1])
		self.linear1 = torch.nn.Linear(14, hid_dim)
		self.linear2 = torch.nn.Linear(hid_dim, hid_dim)
		self.linear3 = torch.nn.Linear(hid_dim, 7)

		self.linear4 = torch.nn.Linear(14+7, 14)

	def forward_obs(self, obs):
		obs = F.relu(self.linear1(obs))
		obs = F.relu(self.linear2(obs))
		obs = self.linear3(obs)
		return obs
		
		
	def forward_act(self, obs, act):
		obs = torch.cat([obs, act],dim=1)
		obs = self.linear4(obs)
		return obs
		
	def forward(self, obs1, obs2, act):

		pred_act = self.forward_obs(obs2-obs1)

		pred_obs1 = self.forward_act(obs1, one_hot_embedding(act, 7))
		pred_obs2 = self.forward_act(obs1, pred_act)

		return pred_act, pred_obs1, pred_obs2
		
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
        obs1 = torch.from_numpy(obs1).unsqueeze(0).float()
        obs2 = torch.from_numpy(obs2).unsqueeze(0).float()
	
        obs1 = (obs1 - test_norm[0]) / test_norm[1]
        obs2 = (obs2 - test_norm[2]) / test_norm[3]
        
        pred_prob, _, _ = test_model(obs1, obs2, torch.zeros(1))
        _, pred = pred_prob.max(1)
        pred = int(pred[0].numpy())
    return pred

if __name__ == '__main__':
    test_initialize()
    obs1 = np.zeros(14, dtype=np.float32)
    obs2 = np.ones(14, dtype=np.float32)
    res = test_query(obs1, obs2)
    print(res)
