"""Simple keeper behavior."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import memory
import core
import commands
import mem_objects
from state_machine import Node, S, T, LoopingStateMachine, C
import UTdebug
import pose
import cfgpose, cfgstiff

class BlockLeft(Node):
    def run(self):
        #commands.stand()
        print('block left')
        UTdebug.log(15, "Blocking left")
            #sit = pose.MyNoBlock()

	pose.ToPose(cfgpose.myrightblock, 1.0).run()

class BlockRight(Node):
    def run(self):
        #commands.stand()
        print('block right')
        UTdebug.log(15, "Blocking right")


class BlockCenter(Node):
    def run(self):
        #commands.stand()
        print('block center')
        UTdebug.log(15, "Blocking right")


class NoBlock(Node):
    def run(self):
        commands.stand()
        print('no block')
        UTdebug.log(15, "Blocking right")





class Blocker(Node):
    def run(self):

        ball = mem_objects.world_objects[core.WO_BALL]
        #commands.setHeadPan(ball.bearing, 0.1)
	
        #print(ball.bearing)
	if ball.seen:
		if ball.right:
			print('right')
			choice = 'right'
		elif ball.left:
			print('left')
                        choice = 'left'
		elif ball.center:
			print('center')
			choice = 'center'
		else:
			print('nothing')
			choice = ''
	else:
		choice = ''
	"""
        if ball.distance < 500:
            UTdebug.log(15, "Ball is close, blocking!")
            if ball.bearing > 30 * core.DEG_T_RAD:
                choice = "left"
            elif ball.bearing < -30 * core.DEG_T_RAD:
                choice = "right"
            else:
                choice = "center"
        """


        #choice = 'nothing'
        self.postSignal(choice)


class Playing(LoopingStateMachine):
    class Stand(Node):
        def run(self):
            commands.setStiffness()
            self.finish()
    class Off(Node):
        def run(self):
            commands.setStiffness(cfgstiff.Zero)
            if self.getTime() > 2.0:
                memory.speech.say("turned off stiffness")
                self.finish()




    def setup(self):
        blocker = Blocker()
        blocks = {"left": BlockLeft(),
                  "right": BlockRight(),
                  "center": BlockCenter() #pose.MyBlockCenter(),#BlockCenter(),
                  #"nothing": pose.MyNoBlock()#NoBlock()
                  }

        for name in blocks:
            sit = pose.MyNoBlock()

            b = blocks[name]
            #self.add_transition(blocker, S(name), b, T(5), sit, T(1), blocker) 
            #self.add_transition(blocker, S(name), b, T(5), sit, T(1), blocker) 

    def run(self):
        print('hao123')
        ps = pose.PoseSequence(cfgpose.myblockcenter, 1.0, cfgpose.sittingPoseNoArms, 1.0)
        ps.run()


