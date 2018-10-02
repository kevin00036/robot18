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

WDIS = 200
MAX_VEL = 0.5

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


def reset_params():
    P = 0
    I = 0
    D = 0
    aP = 0
    aI = 0
    aD = 0
    dP = 0
    dI = 0
    dD = 0
    
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
    else:
        print('9 finish!!!')

class Playing(Task):
    def run(self):
        global ptime, P, I, D, aP, aI, aD, dP, dI, dD, stage_flag, WDIS
        print_flag(stage_flag)
    
    
        if stage_flag == 0:
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
            if not ball.seen:
                commands.setWalkVelocity(0, 0, 0.5)
            else:
                commands.setWalkVelocity(0, 0, 0)
                stage_flag = 1
    
        if stage_flag == 1:
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
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
            #print(aC)
        
            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5
        
            if ball.visionDistance > WDIS:
                commands.setWalkVelocity(C, 0, aC)
            else:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
                if not goal.seen:
                    stage_flag = 2
                else:
                    stage_flag = 3
    
    
        if stage_flag == 2:
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
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
        
            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5
        
            dC = 0.4
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat

            goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
            if not goal.seen:
                commands.setWalkVelocity(C, dC, aC)
            else:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 3
    
    
        if stage_flag == 3:
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
            goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
        
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
        
            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5
        
            ##########################################

            dT = 0
            dV = goal.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP
            dD = dP - dpreP
        
            dC = dP 
        
            if dC > 0.5:
                dC = 0.5
            if dC < -0.5:
                dC = -0.5

            ##########################################
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat
                
            if goal.visionBearing > 0.15 or goal.visionBearing < -0.15 or ball.visionBearing > 0.15 or ball.visionBearing < -0.15:
                commands.setWalkVelocity(C, dC, aC)
                print(dC)
            else:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 4
    
    
    
    
        if stage_flag == 4:
            goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
        
            dtime = self.getTime() - ptime
            ptime = self.getTime()

        
            ##########################################
            T = 1000
            V = goal.distance 
            T2 = WDIS / 2
            V2 = ball.visionDistance
            
            preP = P
            P = min(T - V, T2 - V2)
            print('P = %s' % P)
            I = I + P
            D = P - preP
    
            C = -1.5*P/1000
    
            if C > 0.5:
                C = 0.5
            if C < 0:
                C = 0
    
            aT = 0
            aV = ball.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP/2
        
            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5
        
            ##########################################

            dT = 0
            dV = goal.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP
            dD = dP - dpreP
        
            dC = dP 
        
            if dC > 0.5:
                dC = 0.5
            if dC < -0.5:
                dC = -0.5

            ##########################################
            
            
            if goal.visionBearing > 0.3 or goal.visionBearing < -0.3 or ball.visionBearing > 0.3 or ball.visionBearing < -0.3:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()  
                stage_flag = 1
            elif goal.distance < 1300 and ball.visionDistance < WDIS and abs(goal.visionBearing) < 0.15 and abs(ball.visionBearing) < 0.15:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 5
            else:
                commands.setWalkVelocity(C, 0, aC)

        if stage_flag == 5:
            ball = memory.world_objects.getObjPtr(core.WO_BALL)
            goal = memory.world_objects.getObjPtr(core.WO_OWN_GOAL)
        
            dtime = self.getTime() - ptime
            ptime = self.getTime()
        
        
            ##########################################
            T = 150 / 2
            V = ball.visionDistance  
            
            preP = P
            P = T - V
            I = I + P
            D = P - preP
        
            C = -1.*P/1000
        
            if C > 0.5:
                C = 0.5
            if C < -0.5:
                C = -0.5

            if P > 0:
                C = 0
        
            ##########################################

            dT = -0.4
            dV = ball.visionBearing
            
            dpreP = dP
            dP = dT - dV
            dI = dI + dP
            dD = dP - dpreP
        
            dC = -dP * 1.5
        
            if dC > 0.5:
                dC = 0.5
            if dC < -0.5:
                dC = -0.5

            ##########################################
            aT = 0
            aV = goal.visionBearing
            
            apreP = aP
            aP = aT - aV
            aI = aI + aP
            aD = aP - apreP
        
            aC = -aP * 0.2
        
            if aC > 0.5:
                aC = 0.5
            if aC < -0.5:
                aC = -0.5
            ##########################################
            m = (C * C + dC * dC) ** 0.5
            rat = m / min(m, MAX_VEL)
            C, dC = C * rat, dC * rat
                
            # print('ABS', abs(dV - dT), 'V', V)
            if abs(dV - dT) < 0.05 and V < 150 + 5:
                commands.setWalkVelocity(0, 0, 0)
                reset_params()
                stage_flag = 9
                # print('Good!!!!!')
            else:
                commands.setWalkVelocity(C, dC, aC)
                print(C, dC, aC, 'XD', V, dV, aV)
    
