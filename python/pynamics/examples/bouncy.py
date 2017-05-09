# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
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

error = 1e-4

l1 = Constant('l1',1,system)

#preload1 = Constant('preload1',0*pi/180,system)
m1 = Constant('m1',1e1,system)
m2 = Constant('m2',1e0,system)
k = Constant('k',1e4,system)
l0 = Constant('l0',1,system)
b = Constant('b',1e3,system)
g = Constant('g',9.81,system)

tinitial = 0
tfinal = 10
tstep = .01
t = numpy.r_[tinitial:tfinal:tstep]

x1,x1_d,x1_dd = Differentiable(system,'x1')
x2,x2_d,x2_dd = Differentiable(system,'x2')

initialvalues = {}
initialvalues[x1]=2
initialvalues[x1_d]=0
initialvalues[x2]=1
initialvalues[x2_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
system.set_newtonian(N)

pNA=0*N.x
pm1 = x1*N.y
pm2 = pm1 - x2*N.y

#BodyA = Body('BodyA',A,pm1,m1,IA,system)
Particle1 = Particle(system,pm1,m1,'Particle1')
Particle2 = Particle(system,pm2,m2,'Particle2')

vpm1 = pm1.time_derivative(N,system)
vpm2 = pm2.time_derivative(N,system)

l_ = pm1-pm2
l = (l_.dot(l_))**.5
stretch = l - l0
ul_ = l_*(l**-1)
vl = l_.time_derivative(N,system)

#system.add_spring_force(k,l*ul_,vl)
system.addforce(-k*stretch*ul_,vpm1)
system.addforce(k*stretch*ul_,vpm2)
#system.addforce(k*l*ul_,vpm2)
#system.addforce(-b*vl,vl)
#system.addforce(-b*vl,vl)
#system.addforce(-b*vl,vl)



system.addforcegravity(-g*N.y)

#system.addforcegravity(-g*N.y)
#system.addforcegravity(-g*N.y)


eq1 = [pm2.dot(N.y)-0]
eq1_d=[system.derivative(item) for item in eq1]
eq1_dd=[system.derivative(system.derivative(item)) for item in eq1]
eq = eq1_dd

a = [0-pm2.dot(N.y)]
b = [(item+abs(item)) for item in a]

x1 = Particle1.pCM.dot(N.y)
x2 = Particle2.pCM.dot(N.y)

KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
#func = system.state_space_post_invert(f,ma,eq)
func = system.state_space_post_invert2(f,ma,eq1_dd,eq1_d,eq1,eq_active = b)
print('integrating...')
states=scipy.integrate.odeint(func,ini,t,rtol = error, atol = error, args=(1e6,1e5))
pynamics.toc()
print('calculating outputs..')
output = Output([x1,x2,l],system)
y = output.calc(states)
pynamics.toc()

plt.figure(0)
plt.plot(t,y[:,0])
plt.plot(t,y[:,1])
plt.axis('equal')

plt.figure(1)
plt.plot(t,y[:,2])
plt.axis('equal')
