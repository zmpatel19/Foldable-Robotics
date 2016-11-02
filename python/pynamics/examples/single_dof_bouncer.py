# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics

from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

mA = Constant('mA',1,system)

g = Constant('g',9.81,system)
b = Constant('b',1e0,system)
k = Constant('k',1e10,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

x,x_d,x_dd = Differentiable(system,'x')
y,y_d,y_dd = Differentiable(system,'y')

initialvalues = {}
initialvalues[x]=0
initialvalues[x_d]=.1
initialvalues[y]=.1
initialvalues[y_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')

system.set_newtonian(N)

pNA=0*N.x

pAcm=pNA+x*N.x+y*N.y
vAcm = pAcm.time_derivative(N,system)

ParticleA = Particle(system,pAcm,mA,'ParticleA')

system.addforce(-b*vAcm,vAcm)

stretch = y
stretched1 = (stretch+abs(stretch))/2
stretched2 = -(-stretch+abs(-stretch))/2

#system.add_spring_force(k,(stretched1)*N.y,vAcm) 
system.add_spring_force(k,(stretched2)*N.y,vAcm) 

system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA)-system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func1 = system.createsecondorderfunction2(f,ma)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,KE-PE,x,y],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(y[:,0],y[:,1])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,2])

plt.figure(3)
plt.hold(True)
plt.plot(t,y[:,4])
plt.show()
