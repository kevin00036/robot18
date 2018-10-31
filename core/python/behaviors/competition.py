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
dth = 0.05
class Playing(Task):
    def run(self):
        global headth, dth
        headth = headth + dth
        if headth > 0.5 or headth < -0.5:
            dth = - dth
        #commands.setHeadPan(headth, 0.1)
            

        global flag, remx, remy, counter, target_x, target_y
        robot = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        # commands.setHeadPan(ball.visionBearing, 0.1)
        commands.setHeadPan(headth, 0.1)

        x = robot.loc.x
        y = robot.loc.y
        th = robot.orientation
        """
        print(x, y, th)
        if x < 650 and x > 0 and y < 700 and y > -700:
            print("inside goal box")
        else:
            print("outside goal box")
        """
        mvx = 0
        mvy = 0

        if ball.seen and not ( ball.center or ball.right or ball.left ):
            target_x = 325
            target_y = ball.loc.y / ball.loc.x * 325

        vw = -normAngle(th)
        vw = clip(vw, 0.2)
        dx = target_x - robot.loc.x
        dy = target_y - robot.loc.y
        vx = clip(dx * 0.005, 0.4)
        vy = clip(dy * 0.005, 0.4)

        dis = math.sqrt(dx*dx+dy*dy)
        if dis <= 100:
            vx, vy = 0, 0
        if abs(th) <= 0.1:
            vw = 0
        # if ball.spos - robot.loc.y > 0:
            # vy = 0.3
        # else:
            # vy = -0.3
        """
        if robot.loc.x < 50 or robot.loc.x > 600 or robot.loc.y < -650 or robot.loc.y > 650:
            vy = 0
        """
        commands.setWalkVelocity(vx, vy, vw)
        print(dx, dy, th)
        """
        elif ball.seen and ( ball.center or ball.right or ball.left ):
            if ball.pos > 0:
                ps = pose.ToPose(cfgpose.myblockright, 0.1)
                ps.run()
            else:
                ps = pose.ToPose(cfgpose.myblockleft, 0.1)
                ps.run()
        """
        """
        else:
            mvy = 0

            if x > 600:
                mvx = 600 - x
            if x < 50:
                mvx = 50 - x
        """
        """
        dis = math.sqrt(mvx*mvx+mvy*mvy)
        vabs = min(0.3, dis)

        if abs(dis) > 0.01:
            vx = (+ (mvx) * math.cos(th) + (mvy) * math.sin(th)) / dis * vabs
            vy = (- (mvx) * math.sin(th) + (mvy) * math.cos(th)) / dis * vabs
        else:
            vx = 0
            vy = 0

        #print(round(mvx,3), round(mvy,3))
        #commands.setWalkVelocity(vx, vy, 0)
        """
