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

headth = 0.01
dth = 0.02
class Playing(Task):
    def run(self):
        global headth, dth

        commands.setHeadPan(headth, 0.1)
        
        headth = headth + dth
        if headth > 0.9:
            dth = -dth
        if headth < -0.9:
            dth = -dth
            

        self = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        print(self.loc.x, self.loc.y, self.orientation)

        x = self.loc.x
        y = self.loc.y
        th = self.orientation

        vx = math.sqrt(x*x+y*y)
        vw = normAngle(math.atan2(-y,-x) - th)
        vx = clip(vx, 0.2)
        vw = clip(vw, 0.2)

        commands.setWalkVelocity(vx, 0, vw)
