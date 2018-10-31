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

target_x = 325
target_y = 0

headth = 0
dth = 0.10

state = 0
block_time = 0.
block_start = False
ps = None
curact = ''

class Playing(Task):
    def run(self):
        global state, block_time, block_start, ps, curact
        global headth, dth
        headth = headth + dth
        if headth > 0.5 or headth < -0.5:
            dth = - dth

        global target_x, target_y, headth, dth

        robot = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        # commands.setHeadPan(ball.visionBearing, 0.1)
        commands.setHeadPan(headth, 0.1)

        x = robot.loc.x
        y = robot.loc.y
        th = robot.orientation

        if ball.seen and not ( ball.center or ball.right or ball.left ):
            target_x = 450
            target_y = 2 * ball.loc.y / ball.loc.x * 450
            target_y = clip(target_y, 650)
        elif ball.seen and ( ball.center or ball.right or ball.left ):
            target_x = 450
            target_y = (ball.loc.y - ball.sd.y) / (ball.loc.x - ball.sd.x) * (450 - ball.sd.x) + ball.sd.y
            target_y = clip(target_y, 650)

        vw = -normAngle(th)
        vw = clip(vw, 0.2)
        dx = target_x - robot.loc.x
        dy = target_y - robot.loc.y
        vx = clip(dx * 0.01, 0.4)
        vy = clip(dy * 0.01, 0.4)

        dis = math.sqrt(dx*dx+dy*dy)
        if dis <= 150:
            vx, vy = 0, 0
        if abs(th) <= 0.1:
            vw = 0


        commands.setWalkVelocity(vx, vy, vw)
        print(dx, dy, th)
 
        """
        if state == 0:
            print('nothing')
            if ball.seen and (ball.right or ball.left):
                if ball.right: curact = 'right'
                if ball.left: curact = 'left'
                state = 1
                block_time = self.getTime()
                block_start = True
                commands.setWalkVelocity(0, 0, 0)
            else:
                state = 0
                commands.setWalkVelocity(vx, vy, vw)
                print(dx, dy, th)
        elif state == 1:
            if curact == 'right':
                print('right')
                if block_start:
                    ps = pose.ToPose(cfgpose.myblockright, 0.1)
            elif curact == 'left':
                print('left')
                if block_start:
                    ps = pose.ToPose(cfgpose.myblockleft, 0.1)

            ps.run()
            block_start = False
            if self.getTime() - block_time >= 1:
                ps = pose.ToPose(cfgpose.mynoblock, 1)
                state = 2
                block_time = self.getTime()
        elif state == 2:
            ps.run()

            if self.getTime() - block_time >= 2:
                state = 0
        """
