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
import mem_objects
import math

def normAngle(x):
  while x >= math.pi:
    x = x - 2 * math.pi
  while x < -math.pi: 
    x = x + 2 * math.pi
  return x


class Ready(Task):
    def run(self):
        commands.standStraight()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

def clip(x, mx):
    if x >= mx:
        x = mx
    if x <= -mx:
        x = -mx
    return x



state = 0

headth = 0.01
dth = 0.02

start_time = 0

class Playing(Task):
    def run(self):
        robot = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        global headth, dth, state, start_time

        if state == 0:
            print('stage 0 : flying')
        elif state == 1:
            print('stage 1 : turn')
        elif state == 2:
            print('stage 2 : walk')
            

        if robot.flying:
            state = 0

        if state == 0:
            if not robot.flying:
                state = 1
                start_time = self.getTime()
            
        elif state == 1:

            commands.setWalkVelocity(0, 0, 0.4)
            if self.getTime() - start_time > 15.0:
                state = 2

        elif state == 2:       
            commands.setHeadPan(headth, 0.1)
            headth = headth + dth
            if headth > 1.8:
                dth = -dth
            if headth < -1.8:
                dth = -dth

            x = robot.loc.x
            y = robot.loc.y
            th = robot.orientation

            dis = math.sqrt(x*x+y*y)
            vabs = min(0.3, dis)
            vw = normAngle(math.atan2(-y,-x) - th)
            if abs(vw) < 0.05:
                vw = 0

            vx = (+ (-x) * math.cos(th) + (-y) * math.sin(th)) / dis * vabs
            vy = (- (-x) * math.sin(th) + (-y) * math.cos(th)) / dis * vabs

            if dis > 100:
                vw = clip(vw, 0.2)
            else:
                vw = 0.

            commands.setWalkVelocity(vx, vy, vw)
