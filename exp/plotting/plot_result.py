import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from PIL import Image
import select
import sys

class TimeoutExpired(Exception):
    pass

def input_with_timeout(prompt, timeout):
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [],[], timeout)
    if ready:
        return sys.stdin.readline().rstrip('\n') # expect stdin to be line-buffered
    raise TimeoutExpired

matplotlib.use('Qt4agg')

canvas_size = 250
action = {}
ox, oy = 0, 0
prev_point = []

def to_y(y):
    return canvas_size-y

def to_bear(b):
    return math.pi/2-b
    
def plot_get_act_float(act):
    if np.array_equal(act, [0, 0, 0]): return 'n'
    if np.array_equal(act, [0.3, 0, 0]): return 'f'
    if np.array_equal(act, [-0.3, 0, 0]): return 'b'
    if np.array_equal(act, [0, 0.3, 0]): return 'l'
    if np.array_equal(act, [0, -0.3, 0]): return 'r'
    if np.array_equal(act, [0, 0, 0.3]): return 'tl'
    if np.array_equal(act, [0, 0, -0.3]): return 'tr'

def plot_get_act_int(act):
    return ['n', 'f', 'b', 'l', 'r', 'tl', 'tr'][act]

def get_obj(obj):
    color1 = ['tab:orange','#ffff14','tab:blue','tab:pink','tab:blue','#ffff14','tab:pink']
    color2 = ['tab:orange','tab:blue','#ffff14','tab:blue','tab:pink','tab:pink','#ffff14']

    objs = []
    for i in range(7):
        if obj[2*i] != -1.0:
            objs.append((obj[2*i], obj[2*i+1], color1[i], color2[i]))
    return objs
    #obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
    #[(100, 0.1, 'g'), (500, -0.1, 'b'), (1000, -0.6, 'y')]

def plot_initialize():
    global action, ox, oy, canvas_size
    
    plt.figure(figsize=[7,7])

    prefix = 'plotting/'

    none     = Image.open(prefix + 'none.png')
    forward  = Image.open(prefix + 'forward.png')
    backward = Image.open(prefix + 'backward.png')
    right    = Image.open(prefix + 'right.png')
    left     = Image.open(prefix + 'left.png')
    turnr    = Image.open(prefix + 'turnr.png')
    turnl    = Image.open(prefix + 'turnl.png')


    action['n']     = none.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['f']  = forward.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['b'] = backward.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['r']    = right.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['l']     = left.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['tr']    = turnr.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
    action['tl']    = turnl.resize((canvas_size, canvas_size), Image.ANTIALIAS)  

    # plt.pause(1)

    ox, oy = canvas_size/2, canvas_size - 510/700*canvas_size
    point = plt.scatter(ox,to_y(oy))
    
def plot_update(act, obj):
    global prev_point, canvas_size

    for pt in prev_point:
        pt.remove() 
            
    plt.imshow(action[act])
    point = []
    ss = get_obj(obj)
    for pt in ss:
        distance = pt[0]
        bearing = to_bear(pt[1])
        color = [pt[2], pt[3]]
        
        ratio = canvas_size / 5000
        dx = distance*math.cos(bearing) * ratio
        dy = distance*math.sin(bearing) * ratio
        px = max(0, min(ox + dx, canvas_size))
        py = max(0, min(to_y(oy + dy), canvas_size))
        point.append(plt.scatter(px, py+60*ratio, s = 120, c = color[0], edgecolors = 'k', marker = 's'))
        point.append(plt.scatter(px, py-60*ratio, s = 120, c = color[1], edgecolors = 'k', marker = 's'))
    
    prev_point = point
    # plt.pause(0.01)


def test_main():
    plot_initialize()

    act = plot_get_act_float([0,0,0])
    obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
    plot_update(act,obj)
    
    act = plot_get_act_float([0,0,0.3])
    obj = [-1,-1,-1,-1,-1,-1,1225.69,-0.378,-1,-1,2544.867,-0.125,-1,-1]
    plot_update(act,obj)

    act = plot_get_act_float([0,0,0])
    obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
    plot_update(act,obj)

    plt.pause(3)

def main():
    plot_initialize()
    while True:
        timeout = 1
        try:
            inp = input_with_timeout('', timeout)
        except TimeoutExpired:
            print('Sorry, times up')
            plt.pause(0.001)
            continue

        # inp = input()
        inp = list(map(float, inp.split(',')))
        act = int(inp[0])
        inp = inp[1:]
        print(act, inp)
        pact = plot_get_act_int(act)
        plot_update(pact, inp)
        plt.pause(0.001)
    
if __name__ == '__main__':
    main()

