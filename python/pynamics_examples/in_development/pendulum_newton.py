# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
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

lA = Constant(1,'lA',system)

mA = Constant(1,'mA',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
#x,x_d,x_dd = Differentiable('x',system)
#y,y_d,y_dd = Differentiable('y',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
#initialvalues[x]=1
#initialvalues[y]=0
#initialvalues[x_d]=0
#initialvalues[y_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)

pNA=0*N.x
pAB=pNA+lA*A.x
vAB=pAB.time_derivative(N,system)
#
#pNA2=2*N.x
#pAB2=pNA2+x*N.x+y*N.y
#vAB2=pAB2.time_derivative(N,system)

ParticleA = Particle(pAB,mA,'ParticleA',system)
#ParticleB = Particle(pAB2,mA,'ParticleB',system)

system.addforce(-b*vAB,vAB)
#system.addforce(-b*vAB2,vAB2)
system.addforcegravity(-g*N.y)


f,ma = system.getdynamics()
func = system.state_space_post_invert(f,ma)
q_dd = system.get_q(2)
func = system.solve_f_ma(f,ma,q_dd)
func = func.subs(system.constant_values)
f = sympy.lambdify([qA,qA_d,system.t],func)
func2 = pynamics.integration.build_step(f)
states = pynamics.integration.integrate_newton(f,ini,t)
#func = system.state_space_post_invert(f,ma,eq)
#states=pynamics.integration.integrate_odeint(func,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14, args=({'constants':system.constant_values},))

points = [pNA,pAB,pNA]
points_output = PointsOutput(points,system)
points_output.calc(states)
points_output.animate(fps = 30,lw=2)
