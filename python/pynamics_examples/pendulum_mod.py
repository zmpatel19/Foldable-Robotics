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
from pynamics.output import Output
from pynamics.particle import Particle

import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

lA = Constant(1,'lA',system)
mA = Constant(1,'mA',system)
g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)

x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)

initialvalues = {}
initialvalues[x]=1
initialvalues[y]=0
initialvalues[x_d]=0
initialvalues[y_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
system.set_newtonian(N)

pNA=0*N.x
pAB=pNA+x*N.x+y*N.y
vAB=pAB.time_derivative(N,system)

ParticleA = Particle(pAB,mA,'ParticleA',system)

system.addforce(-b*vAB,vAB)
system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)

KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

v = pAB-pNA
u = (v.dot(v))**.5

eq1 = [(v.dot(v)) - lA**2]
eq1_dd=system.derivative(system.derivative(eq1[0]))
eq = [eq1_dd]

pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func = system.state_space_post_invert(f,ma,eq)
print('integrating...')
states=scipy.integrate.odeint(func,ini,t,args=(),rtol=1e-12,atol=1e-12,hmin=1e-14, args=({'constants':system.constant_values},))
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,KE-PE,x,y],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.plot(y[:,0],y[:,1])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,2])

plt.figure(3)
plt.plot(t,y[:,3])
plt.show()
