from PIL import Image, ImageDraw
import time
import numpy as np
import cv2
from data import *

frame_size = (500, 500)

colors = ['orange', 'blue', 'yellow', 'green', 'darkblue', 'red', 'purple']

image = None
draw = None

def trans_pos(x, y):
    origin = (frame_size[0]//2, frame_size[1]//10)
    width = height = 4000
    outx = int(frame_size[0] / width * (-y) + origin[0])
    outy = frame_size[1] - int(frame_size[1] / height * x + origin[1])
    return outx, outy

def reset_frame():
    global image, draw
    image = Image.new('RGB', frame_size, '#DDD')
    draw = ImageDraw.Draw(image)

    wid = 10
    orig = trans_pos(0, 0)
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

def render_show(delay=200):
    global image, draw
    cv2.imshow('a', np.array(image)[...,::-1])
    cv2.waitKey(delay)

if __name__ == '__main__':
    data = RealData(all_obj=True)
    for dt, obs, act in data:
        reset_frame()
        render_objs(obs)

        render_show()

