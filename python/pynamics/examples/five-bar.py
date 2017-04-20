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

from math import pi,sin,cos

#lA = Constant('lA',1,system)
#lB = Constant('lB',1,system)
#lC = Constant('lC',1,system)


t1 = 30*pi/180
t2 = 100*pi/180
t3 = 70*pi/180
t4 = 60*pi/180
t0 = 2*pi-(t1+t2+t3+t4)

m = Constant('m',1,system)

t0 = Constant('t0',t0,system)
t1 = Constant('t1',t1,system)
t2 = Constant('t2',t2,system)
t3 = Constant('t3',t3,system)
t4 = Constant('t4',t4,system)

g = Constant('g',9.81,system)
b = Constant('b',1e1,system)
k = Constant('k',1e2,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

qA1,qA1_d,qA1_dd = Differentiable(system,'qA1')
qA2,qA2_d,qA2_dd = Differentiable(system,'qA2')
qA3,qA3_d,qA3_dd = Differentiable(system,'qA3')
#qB1,qB1_d,qB1_dd = Differentiable(system,'qB1')
#qB2,qB2_d,qB2_dd = Differentiable(system,'qB2')

initialvalues = {}
initialvalues[qA1]=0*pi/180
initialvalues[qA1_d]=0*pi/180
initialvalues[qA2]=0*pi/180
initialvalues[qA2_d]=0*pi/180
initialvalues[qA3]=0*pi/180
initialvalues[qA3_d]=0*pi/180
#initialvalues[qB1]=0*pi/180
#initialvalues[qB1_d]=0*pi/180
#initialvalues[qB2]=0*pi/180
#initialvalues[qB2_d]=0*pi/180

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A1 = Frame('A1')
A12 = Frame('A12')
A2 = Frame('A2')
A23 = Frame('A23')
A3 = Frame('A3')
A34 = Frame('A34')

#NB1 = Frame('NB1')
#B1 = Frame('B1')
#B12 = Frame('B12')
#B2 = Frame('B2')
#B23 = Frame('B23')

system.set_newtonian(N)

A1.rotate_fixed_axis_directed(N,[1,0,0],qA1,system)
A12.rotate_fixed_axis_directed(A1,[0,0,1],t1,system)
A2.rotate_fixed_axis_directed(A12,[1,0,0],qA2,system)
A23.rotate_fixed_axis_directed(A2,[0,0,1],t2,system)
A3.rotate_fixed_axis_directed(A23,[1,0,0],qA2,system)
A34.rotate_fixed_axis_directed(A3,[0,0,1],t3,system)
#
#NB1.rotate_fixed_axis_directed(N,[0,0,1],-t0,system)
#B1.rotate_fixed_axis_directed(NB1,[1,0,0],-qB1,system)
#B12.rotate_fixed_axis_directed(B1,[0,0,1],-t4,system)
#B2.rotate_fixed_axis_directed(B12,[1,0,0],-qB2,system)
#B23.rotate_fixed_axis_directed(B2,[0,0,1],-t3,system)

pNO = 0*N.x

ParticleA1 = Particle(system,A1.x+A12.x,m,'ParticleA1')
ParticleA2 = Particle(system,A2.x+A23.x,m,'ParticleA2')
ParticleA3 = Particle(system,A3.x+A34.x,m/2,'ParticleA3')
#ParticleB1 = Particle(system,B1.x+B12.x,m,'ParticleA1')
#ParticleB2 = Particle(system,B2.x+B23.x,m/2,'ParticleA2')


wA1 = N.getw_(A1)
wA2 = A12.getw_(A2)
wA3 = A23.getw_(A3)
#wB1 = NB1.getw_(B1)
#wB2 = B2.getw_(B12)

system.addforce(-b*wA1,wA1)
system.addforce(-b*wA2,wA2)
system.addforce(-b*wA3,wA3)
#system.addforce(-b*wB1,wB1)
#system.addforce(-b*wB2,wB2)

system.addforce(1*A1.x,wA1)

system.add_spring_force(k,(qA1)*A1.x,wA1) 
system.add_spring_force(k,(qA2)*A2.x,wA2) 
system.add_spring_force(k,(qA3)*A3.x,wA3) 
#system.add_spring_force(k,(qB1)*B1.x,wB1) 
#system.add_spring_force(k,(qB2)*B2.x,wB2) 

system.addforcegravity(-g*N.y)

#x1 = ParticleA.pCM.dot(N.x)
#y1 = ParticleA.pCM.dot(N.y)
#x2 = ParticleB.pCM.dot(N.x)
#y2 = ParticleB.pCM.dot(N.y)
#x3 = ParticleC.pCM.dot(N.x)
#y3 = ParticleC.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNO) - system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func1 = system.state_space_post_invert(f,ma)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-5,atol=1e-5)
pynamics.toc()
print('calculating outputs..')
output = Output([KE-PE],system)
y = output.calc(states)
pynamics.toc()

plt.figure()
plt.plot(y[:,0])
plt.show()
