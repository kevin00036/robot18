import socket
import pickle

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 34021))
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
            print('Recieved:', data)
        print('Finish')
    except ConnectionResetError:
        print('QQ')
conn.close()

