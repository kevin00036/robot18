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


class Ready(Task):
    def run(self):
        commands.setStiffness()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

ptime = 0.0
P = 0
I = 0
D = 0

aP = 0
aI = 0
aD = 0
class Playing(Task):
    def run(self):
        global ptime, P, I, D, aP, aI, aD
        ball = memory.world_objects.getObjPtr(core.WO_BALL)

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

            C = -1.5*P/1000
            print(C)

            if C > 0.5:
                C = 0.5
            if C < 0:
                C = 0

            """
            """

            aT = 0
            aV = bearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP

            aC = -aP/2
            print(aC)

            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5



            if distance < 150:
                commands.setWalkVelocity(0, 0, 0)
                self.finish()
            else:
                commands.setWalkVelocity(C, 0, aC)
