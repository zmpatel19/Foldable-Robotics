# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.automatic_differentiate=False
#pynamics.script_mode = False
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration
pynamics.script_mode = True

import sympy
#import logging
#pynamics.logger.setLevel(logging.ERROR)
#pynamics.system.logger.setLevel(logging.ERROR)

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

g = Constant(9.81,'g',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

qA,qA_d = Differentiable('qA',limit=2)
qB,qB_d = Differentiable('qB',limit=2)
qC,qC_d = Differentiable('qC',limit=2)

wx,wx_d= Differentiable('wx',ii = 1,limit=3)
wy,wy_d= Differentiable('wy',ii = 1,limit=3)
wz,wz_d= Differentiable('wz',ii = 1,limit=3)

#Differentiable('x')
#Differentiable('y')
#Differentiable('z')

mC = Constant(1,'mC')
Ixx = Constant(2,'Ixx')
Iyy = Constant(3,'Iyy')
Izz = Constant(1,'Izz')

initialvalues = {}
initialvalues[qA]=0*pi/180
# initialvalues[qA_d]=1
initialvalues[qB]=0*pi/180
# initialvalues[qB_d]=1
initialvalues[qC]=0*pi/180
# initialvalues[qC_d]=0

initialvalues[wx]=1.
initialvalues[wy]=1.
initialvalues[wz]=0.

ini = [initialvalues[item] for item in [qA,qB,qC,wx,wy,wz]]

N = Frame('N')
A = Frame('A')
B = Frame('B')
C = Frame('C')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[1,0,0],qA,system)
B.rotate_fixed_axis_directed(A,[0,1,0],qB,system)
C.rotate_fixed_axis_directed(B,[0,0,1],qC,system)

#pCcm=x*N.x+y*N.y+z*N.z
pCcm=0*N.x

# wNC = N.getw_(C)

IC = Dyadic.build(C,Ixx,Iyy,Izz)

w1 = N.getw_(C)
w2 = wx*C.x+wy*C.y+wz*C.z
# C.set_w(N, w2)
N.set_w(C,w2)


eq0 = w1-w2
eq = []
eq.append(eq0.dot(B.x))
eq.append(eq0.dot(B.y))
eq.append(eq0.dot(B.z))

# eq_dd = [system.derivative(item) for item in eq]


import sympy
q_ind = [wx,wy,wz]
q_dep = [qA_d,qB_d,qC_d]


BodyC = Body('BodyC',C,pCcm,mC,IC)

system.addforcegravity(-g*N.y)

points = [1*C.x,0*C.x,1*C.y,0*C.y,1*C.z]

# ang = [wNC.dot(C.x),wNC.dot(C.y),wNC.dot(C.z)]

# func1 = system.state_space_post_invert(f,ma,q_acceleration=[wx_d,wy_d,wz_d],eq_dd=eq,constants = system.constant_values)

# logger.info('solving a = f/m and creating function')
constants = system.constant_values
q_position  = None
q_speed = [wx, wy, wz]
# q_speed = None
q_acceleration = None
# inv_method = 'LU'
# q_dd = q_acceleration=[wx_d,wy_d,wz_d]
f,ma = system.getdynamics(q_speed)

func1 = system.state_space_pre_invert(f,ma,constants = constants,q_position=q_position,q_speed=q_speed,q_acceleration=q_acceleration,q_ind = q_ind,q_dep = q_dep,eq = eq)


states=pynamics.integration.integrate_odeint(func1,ini,t)

# output = Output(ang,system)
# output.calc(states)
# output.plot_time()

po = PointsOutput(points,system,state_variables = [qA,qB,qC,wx,wy,wz])
po.calc(states)
po.animate(fps = 30,lw=2)

so = Output([qA,qB,qC],state_variables = [qA,qB,qC,wx,wy,wz])
so.calc(states)
so.plot_time()