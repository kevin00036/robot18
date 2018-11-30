from sim import Simulator
import torch
import random

sim = Simulator()

data = []
T = 1000

def actgen():
    while True:
        act = random.choice([
            (0., 0., 0.),
            (0.5, 0., 0.),
            (-0.5, 0., 0.),
            (0., 0.5, 0.),
            (0., -0.5, 0.),
            (0., 0., 0.2),
            (0., 0., -0.2),
        ])
        yield act

ag = actgen()
for i in range(T):
    obj = sim.get_obs()
    act = next(ag)
    dt = sim.step(*act)
    data.append((dt, obj, act))

print(data[:10])

