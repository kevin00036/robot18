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

class Ready(Task):
    def run(self):
        commands.setStiffness()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

class Playing(Task):
    def run(self):
        #self = mem_objects.world_objects[core.WO_SELF]
        self = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        #gtObjects->objects_[robotState->WO_SELF]
        #self = memory.robot_state.WO_SELF
        print(self.loc.x, self.loc.y, self.orientation)


