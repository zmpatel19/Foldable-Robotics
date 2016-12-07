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
l2 = Constant('l2',1,system)
l3 = Constant('l3',1,system)
m2 = Constant('m1',1,system)
m1 = Constant('m2',1,system)
g = Constant('g',9.81,system)
b = Constant('b',1e1,system)
k = Constant('k',1e1,system)
#preload1 = Constant('preload1',0*pi/180,system)
k_controller = Constant('k_controller',1e2,system)
q2_command = Constant('q2_command',90*pi/180,system)
#q2_command= Variable('q2_command')

tinitial = 0
tfinal = 1
tstep = .01
t = numpy.r_[tinitial:tfinal:tstep]

x,x_d,x_dd = Differentiable(system,'x')
y,y_d,y_dd = Differentiable(system,'y')
q1,q1_d,q1_dd = Differentiable(system,'q1')
q2,q2_d,q2_dd = Differentiable(system,'q2')

initialvalues = {}
initialvalues[x]=0
initialvalues[y]=1
initialvalues[x_d]=0
initialvalues[y_d]=0
initialvalues[q1]=0
initialvalues[q1_d]=0
initialvalues[q2]=0
initialvalues[q2_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
B = Frame('B')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],q1,system)
B.rotate_fixed_axis_directed(A,[0,0,1],q2,system)

pNA=0*N.x
#pAB=pNA+x*N.x+y*N.y
#vAB=pAB.time_derivative(N,system)

pm1 = x*N.x+y*N.y
vm1 = pm1.time_derivative(N,system)
pm2 = pm1 + l3*B.x
pk1 = pm1-l1*A.x
pk2 = pm1+l2*A.x
vk1 = pk1.time_derivative(N,system)
vk2 = pk2.time_derivative(N,system)

Ixx_A = Constant('Ixx_A',1e-4,system)
Iyy_A = Constant('Iyy_A',1e-4,system)
Izz_A = Constant('Izz_A',1e-4,system)
#Ixx_B = Constant('Ixx_B',6.27600676796613e-07,system)
#Iyy_B = Constant('Iyy_B',1.98358014762822e-06,system)
#Izz_B = Constant('Izz_B',1.98358014762822e-06,system)
#Ixx_C = Constant('Ixx_C',4.39320316677997e-07,system)
#Iyy_C = Constant('Iyy_C',7.9239401855911e-07,system)
#Izz_C = Constant('Izz_C',7.9239401855911e-07,system)
IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
#IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
#IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

BodyA = Body('BodyA',A,pm1,m1,IA,system)
#Particle1 = Particle(system,pm1,m1,'Particle1')
Particle2 = Particle(system,pm2,m2,'Particle2')

s1 = pk1.dot(N.y)*N.y
s2 = pk2.dot(N.y)*N.y
s3 = (q2-q2_command)*A.z
wNA = A.getw_(N)
wNB = B.getw_(N)

#switch1 = 

system.add_spring_force(k,s1,vk1)
system.add_spring_force(k,s2,vk2)
system.add_spring_force(k_controller,s3,wNA)
system.add_spring_force(k_controller,-s3,wNB)

system.addforce(-b*vm1,vm1)

system.addforcegravity(-g*N.y)

#system.addforcegravity(-g*N.y)
#system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = Particle2.pCM.dot(N.x)
y2 = Particle2.pCM.dot(N.y)

KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func = system.state_space_post_invert(f,ma)
print('integrating...')
states=scipy.integrate.odeint(func,ini,t,rtol = error, atol = error, args=(1e4,1e2))
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
plt.plot(t,y[:,3])
plt.show()

#plt.figure(5)
#plt.plot(t,y[:,5:7])
#plt.show()


