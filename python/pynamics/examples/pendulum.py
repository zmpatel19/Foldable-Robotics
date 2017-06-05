# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
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

lA = Constant('lA',1,system)

mA = Constant('mA',1,system)

g = Constant('g',9.81,system)
b = Constant('b',1e0,system)
k = Constant('k',1e1,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant('preload1',0*pi/180,system)

qA,qA_d,qA_dd = Differentiable(system,'qA')

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)

pNA=0*N.x
pAB=pNA+lA*A.x
vAB=pAB.time_derivative(N,system)

ParticleA = Particle(pAB,mA,'ParticleA',system)

system.addforce(-b*vAB,vAB)
system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)

KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func = system.state_space_post_invert(f,ma)
print('integrating...')
states=scipy.integrate.odeint(func,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,KE-PE,qA],system)
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
plt.plot(t,y[:,0])
plt.show()
