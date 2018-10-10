"""Sample behavior."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import core
import memory
import pose
import commands
import cfgpose, cfgstiff
import lights
import math
from task import Task
from state_machine import Node, C, T, StateMachine


class Ready(Task):
    def run(self):
        commands.setStiffness()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

state = 0
block_time = 0.
block_start = False
ps_cen = pose.PoseSequence(cfgpose.myblockcenter, 1.0, cfgpose.sittingPoseNoArms, 1.0)
ps_left = pose.PoseSequence(cfgpose.myblockleft, 1.0, cfgpose.sittingPoseNoArms, 1.0)
ps_right = pose.PoseSequence(cfgpose.myblockright, 1.0, cfgpose.sittingPoseNoArms, 1.0)
ps = None
curact = ''

class Playing(Task):
    def run(self):
        global state, block_time, ps_cen, ps_left, ps_right, block_start, ps, curact

        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        if state == 0:
            print('nothing')
            if ball.seen and (ball.right or ball.left or ball.center):
                if ball.right: curact = 'right'
                if ball.left: curact = 'left'
                if ball.center: curact = 'center'
                state = 1
                block_time = self.getTime()
                block_start = True
            else:
                state = 0
        elif state == 1:
            if curact == 'center':
                print('center')
                if block_start:
                    ps = pose.ToPose(cfgpose.myblockcenter, 0.1)
            elif curact == 'right':
                print('right')
                if block_start:
                    ps = pose.ToPose(cfgpose.myblockright, 0.1)
            elif curact == 'left':
                print('left')
                if block_start:
                    ps = pose.ToPose(cfgpose.myblockleft, 0.1)

            ps.run()
            block_start = False
            if self.getTime() - block_time >= 3.0:
                state = 0

