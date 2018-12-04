import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

canvas_size = 5000
action = {}
ox, oy = 0, 0
prev_point = []

def to_y(y):
	return canvas_size-y

def to_bear(b):
	return math.pi/2-b
	
def get_act(act):
	if np.array_equal(act, [0, 0, 0]): return 'n'
	if np.array_equal(act, [0.3, 0, 0]): return 'f'
	if np.array_equal(act, [-0.3, 0, 0]): return 'b'
	if np.array_equal(act, [0, 0.3, 0]): return 'l'
	if np.array_equal(act, [0, -0.3, 0]): return 'r'
	if np.array_equal(act, [0, 0, 0.3]): return 'tl'
	if np.array_equal(act, [0, 0, -0.3]): return 'tr'

def get_obj(obj):
	color1 = ['tab:orange','#ffff14','tab:blue','tab:pink','tab:blue','#ffff14','tab:pink']
	color2 = ['tab:orange','tab:blue','#ffff14','tab:blue','tab:pink','tab:pink','#ffff14']

	objs = []
	for i in range(7):
		if obj[2*i] is not -1:
			objs.append((obj[2*i], obj[2*i+1], color1[i], color2[i]))
	return objs
	#obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
	#[(100, 0.1, 'g'), (500, -0.1, 'b'), (1000, -0.6, 'y')]
	

def initialize():
	global action, ox, oy, canvas_size
	
	plt.figure(figsize=[10,10])

	none     = Image.open('none.png')
	forward  = Image.open('forward.png')
	backward = Image.open('backward.png')
	right    = Image.open('right.png')
	left     = Image.open('left.png')
	turnr    = Image.open('turnr.png')
	turnl    = Image.open('turnl.png')


	action['n']     = none.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['f']  = forward.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['b'] = backward.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['r']    = right.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['l']     = left.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['tr']    = turnr.resize((canvas_size, canvas_size), Image.ANTIALIAS)  
	action['tl']    = turnl.resize((canvas_size, canvas_size), Image.ANTIALIAS)  

	plt.pause(1)

	ox, oy = canvas_size/2, canvas_size - 510/700*canvas_size
	point = plt.scatter(ox,to_y(oy))


	
	
def update(act, obj):
	global prev_point, canvas_size

	for pt in prev_point:
		pt.remove()	
			
	plt.imshow(action[get_act(act)])
	point = []
	ss = get_obj(obj)
	for pt in ss:
		distance = pt[0]
		bearing = to_bear(pt[1])
		color = [pt[2], pt[3]]
		
		dx = distance*math.cos(bearing)
		dy = distance*math.sin(bearing)
		px = min(ox + dx, canvas_size)
		py = min(to_y(oy + dy), canvas_size)
		point.append(plt.scatter(px, py+60, s = 120, c = color[0], edgecolors = 'k', marker = 's'))
		point.append(plt.scatter(px, py-60, s = 120, c = color[1], edgecolors = 'k', marker = 's'))
	
	prev_point = point
	plt.pause(1)	
	



def main():

	initialize()


	act = [0,0,0]
	obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
	update(act,obj)
	
	act = [0,0,0.3]
	obj = [-1,-1,-1,-1,-1,-1,1225.69,-0.378,-1,-1,2544.867,-0.125,-1,-1]
	update(act,obj)

	act = [0,0,0]
	obj = [-1,-1,-1,-1,-1,-1,2207.794,-0.179,-1,-1,3520.178,-0.011,-1,-1]
	update(act,obj)


		
	plt.pause(10000)



	
if __name__ == '__main__':
	main()

