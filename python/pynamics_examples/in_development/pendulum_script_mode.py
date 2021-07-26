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
import pynamics.integration

#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)
tol = 1e-10
lA = Constant(1)

mA = Constant(1)

g = Constant(9.81)
b_air = Constant(1e-1)
b_joint = Constant(1e-1)
k = Constant(2e1)

Ixx_A = Constant(1)
Iyy_A = Constant(1)
Izz_A = Constant(1)


tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180)

qA,qA_d,qA_dd = Differentiable('qA',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)

pNA=0*N.x
pAB=pNA+lA*A.x
vAB=pAB.time_derivative(N,system)

#ParticleA = Particle(pAB,mA,'ParticleA',system)
IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
BodyA = Body('BodyA',A,pAB,mA,IA,system)

wNA = N.getw_(A)

lab2 = vAB.dot(vAB)
uab = vAB * (1/(lab2**.5+tol))

#squared term
#system.addforce(-b_air*lab2*uab,vAB)
#linear term
system.addforce(-b_air*vAB,vAB)
system.addforce(-b_joint*wNA,wNA)
system.addforcegravity(-g*N.y)
system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 

#x1 = ParticleA.pCM.dot(N.x)
#y1 = ParticleA.pCM.dot(N.y)
x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)

f,ma = system.getdynamics()
func = system.state_space_post_invert(f,ma)
states=pynamics.integration.integrate_odeint(func,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14, args=({'constants':system.constant_values},))

KE = system.get_KE()
PE = system.getPEGravity(pNA) - system.getPESprings()
    
output = Output([x1,y1,KE-PE,qA],system)
y = output.calc(states)

plt.figure(1)
plt.plot(y[:,0],y[:,1])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,2])

plt.figure(3)
plt.plot(t,y[:,0])
plt.show()

import numpy.random

f = f[0].simplify()
ma = ma[0].simplify()

q = y[:,-1].astype(float)
q += numpy.random.rand(len(q))*1e-6
q_d = (q[2:]-q[:-2])/(2*tstep)
q_dd = (q_d[2:]-q_d[:-2])/(2*tstep)


q = q[2:-2]
t = t[2:-2]
q_d = q_d[1:-1]

plt.figure()
plt.plot(t,q)
plt.figure()
plt.plot(t,q_d)
plt.figure()
plt.plot(t,q_dd)


x = numpy.c_[q,numpy.cos(q),q_d]
m = float((ma/qA_dd).subs(system.constant_values))
y = m*q_dd

C = numpy.linalg.solve(x.T.dot(x),x.T.dot(y))
y2 = numpy.r_[[C]].dot(x.T).T

plt.figure()
plt.plot(t,y)
plt.plot(t,y2)

