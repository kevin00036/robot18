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
        # commands.standStraight()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

curElev = 0.0
class Playing(Task):
    def run(self):
        global curElev
        #to detect the blue goal, in ImageProcessor.cpp change if(c==c_ORANGE) to c_BLUE
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        lights.doEyeBalls()
        lights.doPlayingEarLights()
        print(ball.seen)
        if ball.seen:
            bearing = ball.visionBearing
            elev = ball.visionElevation
            #distance = ball.visionDistance   this is the distance to the center of the blue goal
            curElev = 0.8 * curElev + 0.2 * elev
            print(ball.imageCenterX, ball.imageCenterY)
            #print(bearing, elev, distance)
            print(bearing, elev, curElev)
            #if distance>.75:                   walks until 75cm away from blue goal
                #commands.setWalkVelocity(.25,0,0)
                #else: commands.setWalkVelocity(0,0,0)

            if abs(bearing) > math.pi/2:
                if bearing > 0: bearing -= math.pi
                else: bearing += math.pi
                elev = -elev
                
            if abs(bearing) < math.pi/2:
                #commands.setWalkVelocity(0, 0, bearing)    turns at an angle 
                commands.setHeadPan(bearing, 0.5)
            commands.setHeadTilt(-elev / math.pi * 180)
                # commands.setHeadPanTilt(bearing, -curElev / 3.14159 * 180, 0.5)
            # if bearing > 0.05:
                # commands.setHeadPan(0.1, 0.5, True)
            # elif bearing < -0.05:
                # commands.setHeadPan(-0.1, 0.5, True)
            # else:
                # commands.setHeadPan(0.0, 0.5, True)
        # else:
            # commands.setHeadPan(0.0, 0.5, True)


