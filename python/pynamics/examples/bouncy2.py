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
error_tol = 1e-10
l1 = Constant('l1',1,system)

alpha = 1e6
beta = 1e5

#preload1 = Constant('preload1',0*pi/180,system)
m1 = Constant('m1',1e1,system)
m2 = Constant('m2',1e0,system)
k = Constant('k',1e4,system)
l0 = Constant('l0',1,system)
b = Constant('b',1e3,system)
g = Constant('g',9.81,system)


Ixx_A = Constant('Ixx_A',1,system)
Iyy_A = Constant('Iyy_A',1,system)
Izz_A = Constant('Izz_A',1,system)
IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)

tinitial = 0
tfinal = 10
tstep = .01
t = numpy.r_[tinitial:tfinal:tstep]

x1,x1_d,x1_dd = Differentiable(system,'x1')
y1,y1_d,y1_dd = Differentiable(system,'y1')
q1,q1_d,q1_dd = Differentiable(system,'q1')
y2,y2_d,y2_dd = Differentiable(system,'x2')

initialvalues = {}

initialvalues[q1]=0
initialvalues[q1_d]=0

initialvalues[x1]=2
initialvalues[x1_d]=0

initialvalues[y1]=0
initialvalues[y1_d]=0

initialvalues[y2]=1
initialvalues[y2_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],q1,system)

pOrigin = 0*N.x
pm1 = x1*N.x +y1*N.y
pm2 = pm1 - y2*A.y



BodyA = Body('BodyA',A,pm1,m1,IA,system)
Particle2 = Particle(system,pm2,m2,'Particle2')

vpm1 = pm1.time_derivative(N,system)
vpm2 = pm2.time_derivative(N,system)

l_ = pm1-pm2
l = (l_.dot(l_))**.5
stretch = l - l0
ul_ = l_*(l**-1)
vl = l_.time_derivative(N,system)

system.add_spring_force(k,stretch*ul_,vl)
#system.addforce(-k*stretch*ul_,vpm1)
#system.addforce(k*stretch*ul_,vpm2)

system.addforce(-b*l_d*ul_,vpm1)
system.addforce(b*l_d*ul_,vpm2)

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
states=scipy.integrate.odeint(func,ini,t,rtol = error, atol = error, args=(alpha, beta),full_output = 1,mxstep = int(1e5))
states = states[0]
pynamics.toc()
print('calculating outputs..')
output = Output([x1,x2,l, KE-PE],system)
y = output.calc(states)
pynamics.toc()

plt.figure(0)
plt.plot(t,y[:,0])
plt.plot(t,y[:,1])
plt.axis('equal')

plt.figure(1)
plt.plot(t,y[:,2])
plt.axis('equal')

plt.figure(2)
plt.plot(t,y[:,3])
#plt.axis('equal')
