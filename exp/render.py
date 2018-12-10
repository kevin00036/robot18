from PIL import Image, ImageDraw
import time
import numpy as np
import cv2
from data import *

frame_size = (500, 500)

def trans_pos(x, y):
    origin = (frame_size[0]//2, frame_size[1]//10)
    width = height = 4000
    outx = int(frame_size[0] / width * (-y) + origin[0])
    outy = frame_size[1] - int(frame_size[1] / height * x + origin[1])
    return outx, outy

data = RealData(all_obj=True)
for dt, obs, act in data:
    print(obs)
    image = Image.new('RGB', frame_size, '#DDD')
    draw = ImageDraw.Draw(image)
    num = len(obs) // 2

    wid = 10
    orig = trans_pos(0, 0)
    draw.rectangle([(orig[0]-wid,orig[1]-wid), (orig[0]+wid,orig[1]+wid)], fill='red')

    for j in range(num):
        dist, bear = obs[2*j], obs[2*j+1]
        if dist == -1.0:
            continue
        x = dist * np.cos(bear)
        y = dist * np.sin(bear)
        loc = trans_pos(x, y)
        rad = 10
        draw.ellipse([(loc[0]-rad,loc[1]-rad), (loc[0]+rad,loc[1]+rad)], fill='blue')

    # image.show()
    # image.waitKey()
    cv2.imshow('a', np.array(image)[...,::-1])
    cv2.waitKey(200)
    # time.sleep(1)

