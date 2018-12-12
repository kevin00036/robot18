import numpy as np
import random
import math

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
        ball_x = random.randrange(-11250, 1250)
        ball_y = random.randrange(-1750, 750)
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
        

if __name__ == '__main__':
    sim = Simulator()
    for i in range(20):
        print(sim.pos, sim.orien)
        print(sim.get_obs())
        sim.step(0.1, 0.1, 0.1)


