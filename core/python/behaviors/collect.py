"""Sample behavior."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import core
import memory
import pose
import commands
import cfgstiff
import lights
import math
from task import Task
from state_machine import Node, C, T, StateMachine


import sys, tty, termios
from select import select

def getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(sys.stdin.fileno())
            [i, o, e] = select([sys.stdin.fileno()], [], [], 0.1)
            if i:
                ch = sys.stdin.read(1)
            else:
                ch = None

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return ch










class Ready(Task):
    def run(self):
        commands.stand()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

prevtime = 0
f = open('note.txt','a')
class Playing(Task):
    def run(self):
	global prevtime, f

	key = getch()
	if key == None:
		key = 'n'

        time = self.getTime()
	interval = time - prevtime
	prevtime = time


	maxv = 0.3
	maxth = 0.3

	if key == 'n':
		vx, vy, vth = 0, 0, 0
	elif key == 'w':
		vx, vy, vth = maxv, 0, 0
	elif key == 's':
		vx, vy, vth = -maxv, 0, 0
	elif key == 'd':
		vx, vy, vth = 0, -maxv, 0
	elif key == 'a':
		vx, vy, vth = 0, maxv, 0
	elif key == 'e':
		vx, vy, vth = 0, 0, -maxth
	elif key == 'q':
		vx, vy, vth = 0, 0, maxth
	else:
		vx, vy, vth = 0, 0, 0



        commands.setWalkVelocity(vx, vy, vth)


	print(str(round(interval,3))+','+str(vx)+','+str(vy)+','+str(vth),file = f)
	print(str(round(interval,3))+','+str(vx)+','+str(vy)+','+str(vth))
	objids = [
core.WO_BEACON_BALL,
core.WO_AAAAAAAAA,
]
        objs = [memory.world_objects.getObjPtr(oid) for oid in objids]
        memory.world_objects.getObjPtr(core.WO_BEACON_BLUE_YELLOW),
        memory.world_objects.getObjPtr(core.WO_BEACON_YELLOW_BLUE),
        memory.world_objects.getObjPtr(core.WO_BEACON_BLUE_PINK),
        memory.world_objects.getObjPtr(core.WO_BEACON_PINK_BLUE),
        memory.world_objects.getObjPtr(core.WO_BEACON_PINK_YELLOW),
        memory.world_objects.getObjPtr(core.WO_BEACON_YELLOW_PINK)}

	
	if ball.seen:
		print(str(ball.visionDistance)+','+str(ball.visionBearing), file = f)	
		print(str(ball.visionDistance)+','+str(ball.visionBearing))	
	else:
		print('0, 0', file = f)
		print('0, 0')


"""
class Finished(Task):
    def run(self):
        global f
        print('end',file = f)
        commands.setStiffness(cfgstiff.Zero)
        if self.getTime() > 2.0:
            memory.speech.say("turned off stiffness")
            self.finish()
"""
