# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.integrator = 1
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
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
from math import pi
system = System()
pynamics.set_system(__name__,system)

tol=1e-3

lA = Constant(1,'lA',system)

mA = Constant(1,'mA',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
# k = Constant(1e1,'k',system)

Ixx = Constant(1,'Ixx')
Iyy = Constant(1,'Iyy')
Izz = Constant(1,'Izz')

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

# preload1 = Constant(0*pi/180,'preload1',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[x]=0
initialvalues[y]=0
initialvalues[x_d]=0
initialvalues[y_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)

pAcm=x*A.x+y*A.y
pNA=pAcm-lA/2*A.x
pAB=pAcm+lA/2*A.x
vAcm=pAcm.time_derivative(N,system)

IA = Dyadic.build(A,Ixx,Iyy,Izz)


# BodyA = Body('BodyA',A,pAcm,mA,IA)
# ParticleA = Particle(pAB,mA,'ParticleA',system)

system.addforce(-b*vAcm,vAcm)
system.addforcegravity(-g*N.y)


f,ma = system.getdynamics()
func = system.state_space_post_invert(f,ma,constants=system.constant_values)
states=pynamics.integration.integrate(func,ini,t,rtol=tol,atol=tol)

points = [pNA,pAB,pNA]
points_output = PointsOutput(points,system)
points_output.calc(states)
points_output.animate(fps = 30,lw=2)
