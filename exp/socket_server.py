import socket
import pickle
import time
import numpy as np

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
    last_obs = np.zeros(14, dtype=np.float32)

    datagen = socket_datagen()
    for data in datagen:
        assert(len(data) == 1 + 3 + 14) # Time, Action, Observation

        tm = data[0]
        act = np.array(data[1:4], dtype=np.float32)
        obs = np.array(data[4:], dtype=np.float32)

        print(tm, act, obs)

if __name__ == '__main__':
    main()
