"""Sample behavior."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import memory
import pose
import commands
import cfgstiff
from task import Task
from state_machine import Node, C, T, StateMachine

class Ready(Task):
    def run(self):
        commands.standStraight()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()




ptime = 0.0
P = 0
I = 0
D = 0


class Playing(StateMachine):
    class Stand(Node):
        def run(self):
            commands.stand()
            if self.getTime() > 5.0:
                memory.speech.say("playing stand complete")
                self.finish()

    class Walk(Node):
        def run(self):
		global ptime, P, I, D
		ball = mem_objects.world_objects[core.WO_BALL]

		dtime = self.getTime() - ptime
		ptime = self.getTime()

		if ball.seen:
		    bearing = ball.visionBearing
		    distance = ball.visionDistance
		    print(dtime, bearing, distance)

		    T = 0
		    V = distance
		    
		    preP = P
		    P = T - V
		    I = I + P
		    D = P - preP

		    C = -0.5*P/1000
		    print(C)

		    if C > 0.5:
		        C = 0.5
		    if C < 0:
		        C = 0
		    #commands.setWalkVelocity(C, 0, 0)
	       
		
		    if distance < 300:
		        self.finish()

    class Off(Node):
        def run(self):
            commands.setStiffness(cfgstiff.Zero)
            if self.getTime() > 2.0:
                memory.speech.say("turned off stiffness")
                self.finish()

    def setup(self):
        stand = self.Stand()
        walk = self.Walk()
        sit = pose.Sit()
        off = self.Off()
        self.trans(stand, C, walk, T(5.0), sit, C, off)
