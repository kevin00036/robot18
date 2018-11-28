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




import sys
import tty, termios

def getch():
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	return ch




class Ready(Task):
    def run(self):
        commands.stand()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()


class Playing(Task):
    def run(self):
	key = getch()
        time = self.getTime()

        print(time)

	if key == 'w':
                commands.setWalkVelocity(1, 0, 0)
	elif key == 's':
                commands.setWalkVelocity(0, 0, 0)


        
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        print(ball.visionDistance)
