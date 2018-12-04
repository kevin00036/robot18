import socket
import time
import pickle

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('localhost', 34021))
# s.connect(('dum-dums.cs.utexas.edu', 34021))

f = open('data/note1.txt')

for l in f:
    # obj = [
        # 0.2,
        # 0.3, 0.0, 0.0,
        # -1, -1,
        # 3000, 0.7, -1, -1, -1, -1, 2000, -0.5, i, 0.0, -1, -1,
    # ]
    # obj = [float(x) for x in obj]
    obj = list(map(float, l.split(',')))
    print(obj)
    
    bstr = pickle.dumps(obj)

    s.sendall(bstr)
    time.sleep(1)
s.close()
