from sim import Simulator
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

def SimData(T, all_obj=False):
	random.seed(34021501)
	sim = Simulator()
	
	reset_freq = 1000
	
	data = []
	
	def actgen():
		while True:
			repeat = random.randrange(5, 15)
			act = random.choice([
				(0., 0., 0.),
				(0.5, 0., 0.),
				(-0.5, 0., 0.),
				(0., 0.5, 0.),
				(0., -0.5, 0.),
				(0., 0., 0.5),
				(0., 0., -0.5),
			])
			for i in range(repeat):
				yield act
	
	all_obs = []
	position = []
	ag = actgen()
	for i in range(T):
		if i % reset_freq == 0:
			sim.reset()
		_, obs = sim.get_obs()
		if not all_obj:
			obs = obs[2:4]
		act = next(ag)
		dt = sim.step(*act)
		data.append((dt, obs, act))
		all_obs.append(obs)

		p = sim.pos
		# print(p)
		position.append((p[0], p[1]))
	
	np.save('obs.npy',np.array(all_obs))
	
	
	# plt.figure(figsize=[10,10])
	# axes = plt.gca()
	# axes.set_xlim([-1500,1500])
	# axes.set_ylim([-1000,1000])
	
	
	# prev_point = []
	# for p in position:
		# print(p)
		# x = p[0]
		# y = p[1]
		
		# for pt in prev_point:
			# pt.remove()	
		# point = []
		# point.append(plt.scatter(x, y))
		
		# prev_point = point
		# plt.pause(0.2)	
	
	return data

def RealData(path='data/note3.txt', all_obj=False):
    f = open(path, 'r').read().strip().split()
    f = [list(map(float,d.split(','))) for d in f]
    data = [(
        max(0., f[i+1][0] - f[i][0]),
        np.array(f[i][4:]) if all_obj else np.array(f[i][6:8]),
        np.array(f[i][1:4]),
    ) for i in range(len(f)-1)]
    return data
	
	
	
	


	
	
