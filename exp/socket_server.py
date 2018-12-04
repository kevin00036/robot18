import socket
import pickle
import time
import numpy as np
from model_inverse_classify import test_initialize, test_query
# from plotting.plot_result import plot_initialize, plot_update, plot_get_act_float, plot_get_act_int
import subprocess

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
                    break
                # print('Recieved:', data)
                yield data
            # print('Finish')
        except ConnectionResetError:
            print('Connection Lost.')
    conn.close()

def main():
    test_initialize()
    # plot_initialize()
    proc = subprocess.Popen(['python3', 'plotting/plot_result.py'],
                            stdin=subprocess.PIPE,
                            stderr=subprocess.STDOUT,)

    last_obs = np.zeros(14, dtype=np.float32)

    datagen = socket_datagen()
    for data in datagen:
        assert(len(data) == 1 + 3 + 14) # Time, Action, Observation

        tm = data[0]
        act = np.array(data[1:4], dtype=np.float32)
        obs = np.array(data[4:], dtype=np.float32)

        print(act)

        pred_prev_act = test_query(last_obs, obs)
        print('Predicted Action :', pred_prev_act)

        # plot_act = plot_get_act_int(pred_prev_act)
        # plot_update(plot_act, obs)
        print('Send')
        inp_str = ','.join(map(str, [pred_prev_act] + list(obs)))
        proc.stdin.write((inp_str + '\n').encode())
        proc.stdin.flush()

        last_obs = obs

if __name__ == '__main__':
    main()
