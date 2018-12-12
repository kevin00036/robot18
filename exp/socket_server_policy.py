import socket
import pickle
import time
import numpy as np
import random
from model_evaluation import test_initialize, test_query
# from plotting.plot_result import plot_initialize, plot_update, plot_get_act_float, plot_get_act_int
# import subprocess
from data import *
from render import *

def socket_datagen():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('0.0.0.0', 34021))
    s.listen(1)
    while True:
        try:
            conn, addr = s.accept()
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                try:
                    data = pickle.loads(data)
                except:
                    print('Except')
                    break
                print('Recieved:', data)
                act = yield data
                print('Act = ', act)
                data = pickle.dumps(act)
                conn.sendall(data)

            # print('Finish')
        except ConnectionResetError:
            print('Connection Lost.')
    conn.close()

def main():
    test_initialize()
    # proc = subprocess.Popen(['python3', 'plotting/plot_result.py'],
                            # stdin=subprocess.PIPE,
                            # stderr=subprocess.STDOUT,)

    # data = RealData(all_obj=True)
    # step = random.randint(10, 2000)
    step = 1234
    random.seed(34021501)
    data = SimData(20000, all_obj=True)

    goal_obs = data[step][1].astype(np.float32)

    datagen = socket_datagen()
    data = datagen.send(None)

    while True:
        assert(len(data) == 14) # Observation

        obs = np.array(data, dtype=np.float32)

        pred_act = test_query(obs, goal_obs)
        if random.random() < 0.2:
            pred_act = random.randint(0, 6)
        data = datagen.send(pred_act)

        reset_frame()
        render_objs(obs)
        render_objs(goal_obs, rad=7, outline=True)
        render_show(1)

if __name__ == '__main__':
    main()
