# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
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

#import logging
#pynamics.logger.setLevel(logging.ERROR)
#pynamics.system.logger.setLevel(logging.ERROR)

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()

g = Constant(9.81,'g',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

Differentiable('qA',limit=2)
Differentiable('qB',limit=2)
Differentiable('qC',limit=2)

Differentiable('wx',ii = 1,limit=3)
Differentiable('wy',ii = 1,limit=3)
Differentiable('wz',ii = 1,limit=3)

#Differentiable('x')
#Differentiable('y')
#Differentiable('z')

Constant(1,'mC')
Constant(2,'Ixx')
Constant(3,'Iyy')
Constant(1,'Izz')

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=7
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=.2
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=.2

initialvalues[wx]=0
initialvalues[wx_d]=0
initialvalues[wy]=0
initialvalues[wy_d]=0
initialvalues[wz]=0
initialvalues[wz_d]=0

#initialvalues[x]=0
#initialvalues[x_d]=0
#initialvalues[y]=0
#initialvalues[y_d]=0
#initialvalues[z]=0
#initialvalues[z_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

Frame('N')
Frame('A')
Frame('B')
Frame('C')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[1,0,0],qA,system)
B.rotate_fixed_axis_directed(A,[0,1,0],qB,system)
C.rotate_fixed_axis_directed(B,[0,0,1],qC,system)

#pCcm=x*N.x+y*N.y+z*N.z
pCcm=0*N.x
wNC = N.getw_(C)

IC = Dyadic.build(C,Ixx,Iyy,Izz)

w1 = N.getw_(C)
w2 = wx*C.x+wy*C.y+wz*C.z

eq0 = w1-w2
eq = []
eq.append(eq0.dot(N.x))
eq.append(eq0.dot(N.y))
eq.append(eq0.dot(N.z))

eq_dd = [system.derivative(item) for item in eq]

import sympy
ind = sympy.Matrix([wx,wy,wz])
dep = sympy.Matrix([qA_d,qB_d,qC_d])

EQ = sympy.Matrix(eq)
A = EQ.jacobian(ind)
B = EQ.jacobian(dep)
## 
#C = EQ - A*ind - B*dep

dep2 = sympy.simplify(B.solve(-(A),method = 'LU'))
#eq_d = [item.time_derivative() for item in eq]

Body('BodyC',C,pCcm,mC,IC,wNBody=w2)

system.addforcegravity(-g*N.y)

points = [0*N.x,pCcm]

ang = [wNC.dot(C.x),wNC.dot(C.y),wNC.dot(C.z)]

f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,eq_dd)
states=pynamics.integration.integrate_odeint(func1,ini,t, args=({'constants':system.constant_values},))

output = Output(ang,system)
output.calc(states)
output.plot_time()

po = PointsOutput(points,system)
po.calc(states)
