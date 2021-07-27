# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.automatic_differentiate=False
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration
from pynamics.constraint import AccelerationConstraint,KinematicConstraint

import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
import math
system = System()
pynamics.set_system(__name__,system)

g = Constant(9.81,'g',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

# x,x_d,x_dd = Differentiable('x')
# y,y_d,y_dd = Differentiable('y')
# z,z_d,z_dd = Differentiable('z')

qA,qA_d,qA_dd = Differentiable('qA')
qB,qB_d,qB_dd = Differentiable('qB')
qC,qC_d,qC_dd = Differentiable('qC')

wx,wx_d= Differentiable('wx',ii = 1,limit=3)
wy,wy_d= Differentiable('wy',ii = 1,limit=3)
wz,wz_d= Differentiable('wz',ii = 1,limit=3)

mC = Constant(1,'mC')
Ixx = Constant(2,'Ixx')
Iyy = Constant(3,'Iyy')
Izz = Constant(1,'Izz')

initialvalues = {}
initialvalues[qA]=0*math.pi/180
initialvalues[qB]=0*math.pi/180
initialvalues[qC]=0*math.pi/180

# initialvalues[qA_d]=1
# initialvalues[qB_d]=1
# initialvalues[qC_d]=0

initialvalues[wx]=1
initialvalues[wy]=1
initialvalues[wz]=0

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[1,0,0],qA,system)
B.rotate_fixed_axis(A,[0,1,0],qB,system)
C.rotate_fixed_axis(B,[0,0,1],qC,system)

pCcm=0*N.x

IC = Dyadic.build(C,Ixx,Iyy,Izz)

w1 = N.get_w_to(C)
w2 = wx*C.x+wy*C.y+wz*C.z
N.set_w(C,w2)


eq0 = w1-w2
eq0_d = eq0.time_derivative()
eq = []
eq.append(eq0_d.dot(B.x))
eq.append(eq0_d.dot(B.y))
eq.append(eq0_d.dot(B.z))

c = AccelerationConstraint(eq)
# c.linearize(0)
system.add_constraint(c)


eq2 = []
eq2.append(eq0.dot(B.x))
eq2.append(eq0.dot(B.y))
eq2.append(eq0.dot(B.z))
k = KinematicConstraint(eq2)
variables = [qA_d,qB_d,qC_d]
result = k.solve_numeric(variables,[1,1,1],system.constant_values,initialvalues)
initialvalues.update(result)

# for constraint in system.constraints:
    # constraint.solve()

BodyC = Body('BodyC',C,pCcm,mC,IC)

system.addforcegravity(-g*N.y)

# system.addforce(1*C.x+2*C.y+3*C.z,w2)

points = [1*C.x,0*C.x,1*C.y,0*C.y,1*C.z]

f,ma = system.getdynamics()

func1 = system.state_space_pre_invert(f,ma)
# func1 = system.state_space_post_invert(f,ma)

ini = [initialvalues[item] for item in system.get_state_variables()]

states=pynamics.integration.integrate_odeint(func1,ini,t,args=({'constants':system.constant_values},))

po = PointsOutput(points,system)
po.calc(states,t)
po.animate(fps = 30,lw=2)

so = Output([qA,qB,qC])
so.calc(states,t)
so.plot_time()