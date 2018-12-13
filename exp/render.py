from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
import cv2
from data import *

frame_size = (500, 500)

colors = ['orange', 'blue', 'yellow', 'green', 'darkblue', 'red', 'purple']

image = None
draw = None

def discrete_action(act):
    if act[0] > 0: return 1
    if act[0] < 0: return 2
    if act[1] > 0: return 3
    if act[1] < 0: return 4
    if act[2] > 0: return 5
    if act[2] < 0: return 6
    return 0

def trans_pos(x, y):
    origin = (frame_size[0]//2, frame_size[1]//10)
    width = height = 4000
    outx = int(frame_size[0] / width * (-y) + origin[0])
    outy = frame_size[1] - int(frame_size[1] / height * x + origin[1])
    return outx, outy

def reset_frame():
    global image, draw
    image = Image.new('RGB', frame_size, '#999')
    draw = ImageDraw.Draw(image)

    wid = 10
    orig = trans_pos(0, 0)

    view_vertices = [
        orig, 
        (orig[0] - 1000, orig[1] - 1000),
        (orig[0] + 1000, orig[1] - 1000),
    ]
    draw.polygon(view_vertices, fill='#DDD')

    draw.rectangle([(orig[0]-wid,orig[1]-wid), (orig[0]+wid,orig[1]+wid)], fill='black')

def render_objs(obs, rad=10, outline=False):
    global image, draw
    num = len(obs) // 2
    for j in range(num):
        dist, bear = obs[2*j], obs[2*j+1]
        if dist < 0:
            continue
        x = dist * np.cos(bear)
        y = dist * np.sin(bear)
        loc = trans_pos(x, y)
        draw.ellipse([
            (loc[0]-rad,loc[1]-rad), (loc[0]+rad,loc[1]+rad)], 
                     fill=colors[j], outline=('black' if outline else None))

def render_act(act):
    global image, draw
    act_symbols = '·↑↓←→↶↷'
    draw.text((frame_size[0]*0.7, frame_size[1]*0.7), act_symbols[act],
              font=ImageFont.truetype('DejaVuSans.ttf', size=100),
              fill='brown')


def render_show(delay=200):
    global image, draw
    cv2.imshow('a', np.array(image)[...,::-1])
    cv2.waitKey(delay)

def render_save(filename):
    global image, draw
    cv2.imwrite(filename, np.array(image)[...,::-1])

if __name__ == '__main__':
    # data = RealData(all_obj=True)
    data = SimData(20000, all_obj=True)
    for dt, obs, act in data:
        reset_frame()
        render_objs(obs)
        render_act(discrete_action(act))

        render_show()

