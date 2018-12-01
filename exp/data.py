from sim import Simulator
import numpy as np
import random

def SimData(T):
    sim = Simulator()
    random.seed(34021501)

    data = []

    def actgen():
        while True:
            repeat = random.randrange(10)
            for i in range(repeat):
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
        _, obj = sim.get_obs()
        act = next(ag)
        dt = sim.step(*act)
        data.append((dt, obj, act))

    return data

def RealData(path='data/note1.txt'):
    f = open(path, 'r').read().strip().split()
    f = [list(map(float,d.split(','))) for d in f]
    data = [(
        max(0., f[i+1][0] - f[i][0]),
        np.array(f[i][4:6]),
        np.array(f[i][1:4]),
    ) for i in range(len(f)-1)]
    return data
