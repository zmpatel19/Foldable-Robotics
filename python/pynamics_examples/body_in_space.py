# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
# pynamics.automatic_differentiate=False
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration

import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
import math
system = System()
pynamics.set_system(__name__,system)

g = Constant(9.81,'g',system)

tol = 1e-5
tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

qA,qA_d,qA_dd = Differentiable('qA')
qB,qB_d,qB_dd = Differentiable('qB')
qC,qC_d,qC_dd = Differentiable('qC')


mC = Constant(1,'mC')
Ixx = Constant(2,'Ixx')
Iyy = Constant(3,'Iyy')
Izz = Constant(1,'Izz')

initialvalues = {}
initialvalues[qA]=0*math.pi/180
initialvalues[qA_d]=1
initialvalues[qB]=0*math.pi/180
initialvalues[qB_d]=1
initialvalues[qC]=0*math.pi/180
initialvalues[qC_d]=0

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[1,0,0],qA,system)
B.rotate_fixed_axis(A,[0,1,0],qB,system)
C.rotate_fixed_axis(B,[0,0,1],qC,system)

pCcm=0*N.x
w1 = N.get_w_to(C)


IC = Dyadic.build(C,Ixx,Iyy,Izz)

BodyC = Body('BodyC',C,pCcm,mC,IC)

system.addforcegravity(-g*N.y)

# system.addforce(1*C.x+2*C.y+3*C.z,w1)

points = [1*C.x,0*C.x,1*C.y,0*C.y,1*C.z]

f,ma = system.getdynamics()

# func1 = system.state_space_pre_invert(f,ma)
func1 = system.state_space_post_invert(f,ma)

ini = [initialvalues[item] for item in system.get_state_variables()]

states=pynamics.integration.integrate_odeint(func1,ini,t,rtol = tol, atol=tol,args=({'constants':system.constant_values},))

po = PointsOutput(points,system)
po.calc(states,t)
po.animate(fps = 30,lw=2)

so = Output([qA,qB,qC])
so.calc(states,t)
so.plot_time()