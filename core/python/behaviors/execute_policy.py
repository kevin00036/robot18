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

import socket
import pickle

import sys, tty, termios
from select import select

print('Start Connect Socket')
py_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# py_socket.connect(('dum-dums.cs.utexas.edu', 34021))
py_socket.connect(('128.83.252.101', 34021))
print('Socket Connected')

class Ready(Task):
    def run(self):
        commands.stand()
        if self.getTime() > 5.0:
            memory.speech.say("ready to play")
            self.finish()

class Playing(Task):
    def run(self):
        global prevtime

        commands.setHeadPan(0.0, 0.5)
        commands.setHeadTilt(-10.0)

        time = self.getTime()
        objids = [core.WO_BALL,
                  core.WO_BEACON_BLUE_YELLOW,
                  core.WO_BEACON_YELLOW_BLUE,
                  core.WO_BEACON_BLUE_PINK,
                  core.WO_BEACON_PINK_BLUE,
                  core.WO_BEACON_PINK_YELLOW,
                  core.WO_BEACON_YELLOW_PINK]
        objs = [memory.world_objects.getObjPtr(oid) for oid in objids]

        # Send socket
        objarr = []
        for obj in objs:
            if obj.seen:
                objarr.extend([obj.visionDistance, obj.visionBearing])
            else:
                objarr.extend([-1., -1.])
        objarr = [float(x) for x in objarr]
        print(objarr)
        
        bstr = pickle.dumps(objarr)
        py_socket.sendall(bstr)

        bstr = py_socket.recv(1024)
        act = pickle.loads(bstr)

        maxv = 0.5
        maxth = 0.2

        if act == 0:
            vx, vy, vth = 0, 0, 0
        elif act == 1:
            vx, vy, vth = maxv, 0, 0
        elif act == 2:
            vx, vy, vth = -maxv, 0, 0
        elif act == 3:
            vx, vy, vth = 0, maxv, 0
        elif act == 4:
            vx, vy, vth = 0, -maxv, 0
        elif act == 5:
            vx, vy, vth = 0, 0, maxth
        elif act == 6:
            vx, vy, vth = 0, 0, -maxth
        else:
            vx, vy, vth = 0, 0, 0

        commands.setWalkVelocity(vx, vy, vth)

