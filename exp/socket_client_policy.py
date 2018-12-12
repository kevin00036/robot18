import socket
import time
import pickle
import random
from data import *

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 34021))
# s.connect(('dum-dums.cs.utexas.edu', 34021))

mode = 'sim'
# mode = 'real'

if mode == 'real':
    data = RealData(all_obj=True)
else:
    random.seed(34021501)
    sim = Simulator()
    # data = SimData(20000, all_obj=True)

while True:
    _, obs = sim.get_obs()
    arr = list(obs)
    print(arr)
    bstr = pickle.dumps(arr)
    s.sendall(bstr)

    bstr = s.recv(1024)
    act = pickle.loads(bstr)
    print('Act', act)
    sim.step(*get_actvec(act))
    time.sleep(0.1)
    # time.sleep(1)



s.close()
