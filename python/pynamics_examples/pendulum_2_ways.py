# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
sympy.init_printing(pretty_print=False)

import pynamics
pynamics.integrator = 0
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
from pynamics.constraint import AccelerationConstraint
import pynamics.integration

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(1,'lA',system)

mA = Constant(1,'mA',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)

preload1 = Constant(0*pi/180,'preload1',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[x]=1
initialvalues[y]=0
initialvalues[x_d]=0
initialvalues[y_d]=0

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)

pNA=0*N.x
pAB=pNA+lA*A.x
vAB=pAB.time_derivative(N,system)

pNA2=2*N.x
pAB2=pNA2+x*N.x+y*N.y
vAB2=pAB2.time_derivative(N,system)

ParticleA = Particle(pAB,mA,'ParticleA',system)
ParticleB = Particle(pAB2,mA,'ParticleB',system)

system.addforce(-b*vAB,vAB)
system.addforce(-b*vAB2,vAB2)
system.addforcegravity(-g*N.y)


v = pAB2-pNA2
u = (v.dot(v))**.5

eq1 = v.dot(v) - lA**2
eq1_d=system.derivative(eq1)
eq1_dd=system.derivative(eq1_d)
eq = [eq1_dd]

constraint = AccelerationConstraint([eq1_dd])
system.add_constraint(constraint)

tol = 1e-4

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

f,ma = system.getdynamics()
func = system.state_space_post_invert(f,ma,constants=system.constant_values)

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]


states=pynamics.integration.integrate(func,ini,t,rtol=tol,atol=tol)

points = [pNA,pAB,pNA,pNA2,pAB2]
points_output = PointsOutput(points,system)
points_output.calc(states,t)
points_output.plot_time()
#points_output.animate(fps = 30,lw=2)
