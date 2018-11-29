import numpy as np

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
        self.objects = [
            Point(2000, 1250),
            Point(-1000, 0),
        ]

        self.pos = Point(0, 0)
        self.vel = Point(0, 0)
        self.orien = 0.0
        self.angvel = 0.0
        self.dt = 0.1
        self.fov = 45 * (np.pi / 180.)
        self.cur_time = 0.0

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
        return np.array(obs)

    def step(self, ax, ay, alpha):
        ax = clip(ax, -1., 1.) * 1000
        ay = clip(ay, -1., 1.) * 1000
        alpha = clip(alpha, -1., 1.)

        dt = self.dt

        self.vel[0] = clip(self.vel[0] + ax * dt, -100., 100.)
        self.vel[1] = clip(self.vel[1] + ay * dt, -100., 100.)
        self.angvel = clip(self.angvel + alpha, -1., 1.)

        self.pos += self.vel * dt
        self.orien = reg_angle(self.orien + self.angvel * dt) 
        

if __name__ == '__main__':
    sim = Simulator()
    for i in range(10):
        print(sim.pos, sim.orien)
        print(sim.get_obs())
        sim.step(0.1, 0.1, 0.1)

    
