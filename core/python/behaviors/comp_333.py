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


class Ready(Task):
    def run(self):
        commands.setStiffness()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

WDIS = 200
MAX_VEL = 0.4

ptime = 0.0
P = 0
I = 0
D = 0

aP = 0
aI = 0
aD = 0

dP = 0
dI = 0
dD = 0

stage_flag = 0
kick_frame = 0
finish_time = 0
align_time = 0


def reset_params():
    global ptime, P, I, D, aP, aI, aD, dP, dI, dD, stage_flag, WDIS, kick_frame, finish_time, align_time
    P = 0
    I = 0
    D = 0
    aP = 0
    aI = 0
    aD = 0
    dP = 0
    dI = 0
    dD = 0

def normAngle(x):
    while x > math.pi:
        x -= 2 * math.pi
    while x <= -math.pi:
        x += 2 * math.pi
    return x
    
def print_flag(stage_flag):
    if stage_flag == 0:
        print('0 find ball')
    elif stage_flag == 1:
        print('1 walk ball')
    elif stage_flag == 2:
        print('2 find goal')
    elif stage_flag == 3:
        print('3 face goal')
    elif stage_flag == 4:
        print('4 walk goal')
    elif stage_flag == 5:
        print('5 align')
    elif stage_flag == 6:
        print('6 kick', kick_frame)
    elif stage_flag == 9:
        print('9 finish!!!')
    else:
        print('Stage Flag Is OMiMAF', stage_flag)

def clip(x, mx):
    if x >= mx:
        x = mx
    if x <= -mx:
        x = -mx
    return x

class Playing(Task):
    def run(self):
        robot = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
        global ptime, P, I, D, aP, aI, aD, dP, dI, dD, stage_flag, WDIS, kick_frame, finish_time, align_time
        print_flag(stage_flag)
    
        commands.setStiffness()


        x = robot.loc.x
        y = robot.loc.y
        th = robot.orientation
        goal.visionDistance = math.sqrt(x*x+y*y)
        if not goal.seen:
            goal.visionBearing = normAngle(math.atan2(-y, -x) - th)
        goal.seen = True
        print('Goal VDis', goal.visionDistance, 'VBear', goal.visionBearing)
        print('  Ball VDis', ball.visionDistance, 'VBear', ball.visionBearing)

        # Find ball
        if stage_flag == 0:
            if not ball.seen:
                commands.setWalkVelocity(0, 0, MAX_VEL)
            else:
                # commands.setWalkVelocity(0, 0, 0)
                stage_flag = 1
    
        # Walk ball
        if stage_flag == 1:
            dtime = self.getTime() - ptime
            ptime = self.getTime()
        
            
            ##########################################
            T = 0
            V = ball.visionDistance 
            
            preP = P
            P = T - V
            I = I + P
            D = P - preP
        
            C = -1.5*P/1000
            C = clip(C, MAX_VEL)
        
            ##########################################
        
            aT = 0
            aV = ball.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP/2
            aC = clip(aC, MAX_VEL)
        
            # if not ball.seen:
                # stage_flag = 0
            if ball.visionDistance > WDIS:
                commands.setWalkVelocity(C, 0, aC)
            else:
                # commands.setWalkVelocity(0, 0, 0)
                reset_params()
                if not goal.seen:
                    stage_flag = 2
                else:
                    stage_flag = 3
    
    
        # Find goal
        if stage_flag == 2:
            dtime = self.getTime() - ptime
            ptime = self.getTime()


            ##########################################
            T = WDIS
            V = ball.visionDistance  
            
            preP = P
            P = T - V
            I = I + P
            D = P - preP
        
            C = -1.5*P/1000
            C = clip(C, MAX_VEL)
            C = clip(C, MAX_VEL)

            if C > 0.5:
                C = 0.5
            if C < -0.5:
                C = -0.5
        
            ##########################################
        
            aT = 0
            aV = ball.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP/2
            aC = clip(aC, MAX_VEL)
        
            dC = MAX_VEL
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat

            # if not ball.seen:
                # stage_flag = 0
            if not goal.seen:
                commands.setWalkVelocity(C, dC, aC)
            else:
                # commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 3
    
    
        # Face goal
        if stage_flag == 3:
            dtime = self.getTime() - ptime
            ptime = self.getTime()
        
        
            ##########################################
            T = WDIS
            V = ball.visionDistance  
            
            preP = P
            P = T - V
            I = I + P
            D = P - preP
        
            C = -1.5*P/1000
            C = clip(C, MAX_VEL)
        
            ##########################################
        
            aT = 0
            aV = ball.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP/2
            aC = clip(aC, MAX_VEL)
        
            ##########################################

            dT = 0
            dV = goal.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP
            dD = dP - dpreP
        
            dC = dP 
            dC = clip(dC, MAX_VEL)

            ##########################################
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat
                
            # if not ball.seen:
                # stage_flag = 0
            if not goal.seen:
                stage_flag = 2
            elif abs(goal.visionBearing) > 0.15 or abs(ball.visionBearing) > 0.15:
                commands.setWalkVelocity(C, dC, aC)
                print(dC)
            else:
                # commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 4
    
    
        # Walk goal
        if stage_flag == 4:
            dtime = self.getTime() - ptime
            ptime = self.getTime()

        
            ##########################################
            T = 1300
            V = goal.distance 
            T2 = WDIS / 2
            V2 = ball.visionDistance
            
            preP = P
            P = min(T - V, T2 - V2)
            print('P = %s' % P)
            I = I + P
            D = P - preP
    
            C = -1.5*P/1000
            C = clip(C, MAX_VEL)
    
            aT = 0
            aV = ball.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP/2
            aC = clip(aC, MAX_VEL)
        
            ##########################################

            dT = 0
            dV = goal.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP
            dD = dP - dpreP
        
            dC = dP 
            dC = clip(dC, MAX_VEL)

            ##########################################
            
            
            if abs(goal.visionBearing) > 0.4 or abs(ball.visionBearing) > 0.3:
                # commands.setWalkVelocity(0, 0, 0)
                reset_params()  
                stage_flag = 1
            elif goal.distance < T + 100 and ball.visionDistance < WDIS and abs(goal.visionBearing) < 0.3 and abs(ball.visionBearing) < 0.15:
                # commands.setWalkVelocity(0, 0, 0)
                reset_params()
                align_time = self.getTime()
                stage_flag = 5
            else:
                commands.setWalkVelocity(C, 0, aC)

        # Align
        if stage_flag == 5:
        
            dtime = self.getTime() - ptime
            ptime = self.getTime()
        
        
            ##########################################
            T = 150 / 2
            V = ball.visionDistance  
            
            preP = P
            P = T - V
            I = I + P * dtime
            D = P - preP
        
            C = (-1. * P - I * 0.1)/1000
            C = clip(C, 0.3)

            if P > 0:
                C = 0
        
            ##########################################

            dT = -0.3
            dV = ball.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP * dtime
            dD = dP - dpreP
        
            dC = -dP * 1.5 - dI * 0.1
            dC = clip(dC, 0.3)

            ##########################################
            aT = 0
            aV = goal.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP * 0.4
            aC = clip(aC, MAX_VEL)

            ##########################################
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat
                
            # print('ABS', abs(dV - dT), 'V', V)
            if (abs(dV - dT) < 0.05 and V < 130) or (self.getTime() - align_time >= 15):
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 6
                kick_frame = 0
            else:
                commands.setWalkVelocity(C, dC, aC)
                print(C, dC, aC, 'XD\n', V, dV, aV, dI)
    
        # Kick
        if stage_flag == 6:
            if kick_frame <= 3:
                memory.walk_request.noWalk()
                memory.kick_request.setFwdKick()
            if kick_frame > 10 and not memory.kick_request.kick_running_:
                stage_flag = 9
                finish_time = self.getTime()
            kick_frame += 1

        # Finish
        if stage_flag == 9:
            commands.stand()
            if self.getTime() - finish_time >= 7:
                stage_flag = 0
                # self.finish()

        # head turn
        T_H = 5
        if int(self.getTime() / T_H) % 4 == 0:
            tm = math.fmod(self.getTime() + T_H/4., T_H)
            hpan = -1.8 + 3.6 * (min(tm, T_H-tm) / T_H * 2)
            commands.setHeadPan(hpan, 0.1)
            commands.setWalkVelocity(0, 0, 0)


target_x = 325
target_y = 0

headth = 0
dth = 0.03

state = 0
block_time = 0.
block_start = False
ps = None
curact = ''
track_th = 0.

seenstate = False
class Penalised(Task):
    def run(self):
        global state, block_time, block_start, ps, curact
        global target_x, target_y, headth, dth, track_th

        robot = memory.world_objects.getObjPtr(memory.robot_state.WO_SELF)
        ball = memory.world_objects.getObjPtr(core.WO_BALL)
        # commands.setHeadPan(ball.visionBearing, 0.1)

        if ball.seen:
            track_th = ball.visionBearing
        else:
            track_th *= 0.97

        lb = max(track_th - 0.7, -1.8)
        rb = min(track_th + 0.7, 1.8)

        if headth > rb:
            headth = rb
        if headth < lb:
            headth = lb
        if headth >= rb or headth <= lb:
            dth = - dth
        headth = headth + dth
        commands.setHeadPan(headth, 0.1)
        # print(lb, rb, dth, headth)
            
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
        vx = clip(dx * 0.01, 0.3)
        vy = clip(dy * 0.01, 0.3)

        dis = math.sqrt(dx*dx+dy*dy)
        if dis <= 50:
            vx, vy = 0, 0
        if abs(th) <= 0.15:
            vw = 0

        commands.setWalkVelocity(vx, vy, vw)
        print(dx, dy, th)
 
