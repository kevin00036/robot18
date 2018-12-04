import socket
import time
import pickle
import random
from data import SimData, RealData

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 34021))
# s.connect(('dum-dums.cs.utexas.edu', 34021))

mode = 'sim'
# mode = 'real'

# if mode == 'real':
    # f = open('data/note1.txt')

    # for l in f:
        # obj = list(map(float, l.split(',')))
        # print(obj)
        
        # bstr = pickle.dumps(obj)
        # s.sendall(bstr)
        # time.sleep(1)

if mode == 'real':
    data = RealData(all_obj=True)
else:
    random.seed(34021501)
    data = SimData(10000, all_obj=True)

for d in data:
    dt, obs, act = d
    # arr = [dt] + [a*0.6 for a in act] + list(obs)
    arr = [dt] + list(act) + list(obs)
    print(arr)
    bstr = pickle.dumps(arr)
    s.sendall(bstr)
    time.sleep(0.1)
    # time.sleep(1)



s.close()
