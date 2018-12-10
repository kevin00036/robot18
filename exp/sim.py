import numpy as np
import random
import math
import torch
import torch.nn.functional as F

model_path = 'models/inverse_classify_sim.mdl'
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
		
		

def Point(x, y):
    return np.array([x, y], dtype=float)

def clip(x, lb, rb):
    return max(lb, min(rb, x))

def reg_angle(th):
    while th >= np.pi:
        th -= 2 * np.pi
    while th < -np.pi:
        th += 2 * np.pi
    return th

class Simulator:
	def __init__(self):
		self.reset()
	
	def reset(self):
		ball_x = random.randrange(-1250, 1250)
		ball_y = random.randrange(-750, 750)
		self.objects = [
			Point(ball_x, ball_y),
			Point(1500, 1000),
			Point(1500, -1000),
			Point(0, 1000),
			Point(0, -1000),
			Point(-1500, 1000),
			Point(-1500, -1000),
		]
	
		self.pos = Point(0, 0)
		self.vel = Point(0, 0)
		self.orien = 0.0
		self.angvel = 0.0
		self.dt = 0.2
		self.fov = 45 * (np.pi / 180.)
		self.cur_time = 0.0
		self.max_acc = 300.
		self.max_ang_acc = 1.
	
	def get_obs(self):
		obs = []
		for obj in self.objects:
			dpos = obj - self.pos
			dis = np.linalg.norm(dpos)
			bear = reg_angle(np.arctan2(dpos[1], dpos[0]) - self.orien)
			if abs(bear) > self.fov:
				dis = -1
				bear = -1
			obs.extend([dis, bear])
		return self.cur_time, np.array(obs)
	
	def step(self, actx, acty, actalpha):
		rel_vx = actx * 300.
		rel_vy = acty * 300.
		tar_angvel = actalpha
	
		cs = math.cos(self.orien)
		sn = math.sin(self.orien)
		tar_vx = cs * rel_vx - sn * rel_vy
		tar_vy = sn * rel_vx + cs * rel_vy
		
		dvx = tar_vx - self.vel[0]
		dvy = tar_vy - self.vel[1]
		dvang = tar_angvel - self.angvel
	
		dv_abs = (dvx ** 2 + dvy ** 2) ** 0.5 + 1e-5
		acc_abs = self.max_acc * min(1., dv_abs / 50.)
		
		ax = (dvx / dv_abs) * acc_abs
		ay = (dvy / dv_abs) * acc_abs
		alpha = self.max_ang_acc * clip(dvang / 0.2, -1., 1.)
	
		# print(self.vel, self.angvel, '|', actx, acty, actalpha, '|', ax, ay, alpha)
	
		dt = self.dt
	
		self.vel[0] = clip(self.vel[0] + ax * dt, -300., 300.)
		self.vel[1] = clip(self.vel[1] + ay * dt, -300., 300.)
		self.angvel = clip(self.angvel + alpha * dt, -1., 1.)
	
		self.pos += self.vel * dt
		self.orien = reg_angle(self.orien + self.angvel * dt) 
	
		self.cur_time += dt
		return dt
		
	def fly(self, state):
		self.pos = state[0]
		self.vel = state[1]
		self.orien = state[2]
		self.angvel = state[3]	
		
###############################################################
		
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

	
	
def action_mapping(sim, act):
	if act == 0:  sim.step(0, 0, 0)
	if act == 1 : sim.step(0.5, 0, 0) 
	if act == 2 : sim.step(-0.5, 0, 0)
	if act == 3 : sim.step(0, 0.5, 0) 
	if act == 4 : sim.step(0, -0.5, 0)
	if act == 5 : sim.step(0, 0, 0.2)
	if act == 6 : sim.step(0, 0, -0.2)

	
##############
"""
        self.pos = Point(0, 0)
        self.vel = Point(0, 0)
        self.orien = 0.0
        self.angvel = 0.0
"""		
##############

def L2(a, b):
	return np.sum((a - b)**2) ** 0.5


if __name__ == '__main__':
	sim = Simulator()
	
	"""
	for i in range(20):
		print(sim.pos, sim.orien)
		print(sim.get_obs())
		sim.step(0.1, 0.1, 0.1)
	"""
	
	
	test_initialize()
	
	
	
	K = 5
	T = K
	
	start = sim.get_obs()
	state = (sim.pos.copy(), sim.vel.copy(), sim.orien, sim.angvel)	
	print(state)
	for i in range(T//K):
		act = random.randint(0, 6)
		print(act)
		for j in range(K):
			action_mapping(sim, act)
	end = sim.get_obs()

	print(state)	
	state2 = (sim.pos.copy(), sim.vel.copy(), sim.orien, sim.angvel)		
	print(state2)
	
	
	sim.fly(state)
	

	
	start = sim.get_obs()
	real_acts = []
	for i in range(T):
		act = test_query(start[1], end[1])
		action_mapping(sim, act)
		start = sim.get_obs()
		real_acts.append(act)
	pred_end = start
	
	state3 = (sim.pos.copy(), sim.vel.copy(), sim.orien, sim.angvel)		
	print(state3)
	print(real_acts)
	
	print(L2(end[1], pred_end[1]))
	print([L2(state2[i], state3[i]) for i in range(4)])

	