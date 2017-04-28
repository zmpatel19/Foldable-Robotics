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
tol = 1e-10
lA = Constant('lA',0.05699,system)

mA = Constant('mA',0.0054531675,system)

g = Constant('g',9.81,system)
b_air = Constant('b_air',0,system)
b_joint = Constant('b_joint',0.000023855338,system)
k = Constant('k',0.016705166667,system)

Ixx_A = Constant('Ixx_A',1,system)
Iyy_A = Constant('Iyy_A',1,system)
Izz_A = Constant('Izz_A',0.00000428720184,system)


tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant('preload1',0*pi/180,system)

qA,qA_d,qA_dd = Differentiable(system,'qA')

initialvalues = {}
initialvalues[qA]=10*pi/180
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

#ParticleA = Particle(system,pAB,mA,'ParticleA')
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
system.addforcegravity(-g*N.x)
system.add_spring_force(k,(qA-preload1)*N.z,wNA) 

#x1 = ParticleA.pCM.dot(N.x)
#y1 = ParticleA.pCM.dot(N.y)
x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)

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

#plt.figure(1)
#plt.plot(y[:,0],y[:,1])
#plt.axis('equal')
plt.figure(1)
plt.plot(t,y[:,0])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,1]*180/pi)

plt.figure(3)
plt.plot(y[:,2]*180/pi)

plt.figure(4)
plt.plot(t,y[:,-1]*180/pi)
plt.show()


#
#import numpy.random
#
#f = f[0].simplify()
#ma = ma[0].simplify()
#
q = y[:,-1].astype(float)
q += numpy.random.rand(len(q))*1e-6
q_d = (q[2:]-q[:-2])/(2*tstep)
q_dd = (q_d[2:]-q_d[:-2])/(2*tstep)
#
#
q = q[2:-2]
t = t[2:-2]
q_d = q_d[1:-1]
#
#plt.figure()
#plt.plot(t,q)
#plt.figure()
#plt.plot(t,q_d)
#plt.figure()
#plt.plot(t,q_dd)
#
#
#x = numpy.c_[q,numpy.cos(q),q_d]
#m = float((ma/qA_dd).subs(system.constants))
#y = m*q_dd
#
#C = numpy.linalg.solve(x.T.dot(x),x.T.dot(y))
#y2 = numpy.r_[[C]].dot(x.T).T
#
#plt.figure()
#plt.plot(t,y)
#plt.plot(t,y2)

lines = []
for item in zip(t,q,q_d,q_dd):
    lines.append('{0:0.5e},{1:0.5e},{2:0.5e},{3:0.5e}\n'.format(*item))

with open('output.csv','w') as f:
    f.writelines(lines)