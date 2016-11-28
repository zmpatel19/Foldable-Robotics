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

lA = Constant('lA',.075,system)
lB = Constant('lB',.02,system)

mA = Constant('mA',.01,system)
mB = Constant('mB',.1,system)

g = Constant('g',9.81,system)

tinitial = 0
tfinal = 10
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

Ixx_A = Constant('Ixx_A',.05/100/100,system)
Iyy_A = Constant('Iyy_A',.05/100/100,system)
Izz_A = Constant('Izz_A',.05/100/100,system)
Ixx_B = Constant('Ixx_B',2.5/100/100,system)
Iyy_B = Constant('Iyy_B',.5/100/100,system)
Izz_B = Constant('Izz_B',2/100/100,system)

qA,qA_d,qA_dd = Differentiable(system,'qA')
qB,qB_d,qB_dd = Differentiable(system,'qB')

initialvalues = {}
initialvalues[qA]=45*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=.5*pi/180
initialvalues[qB_d]=0*pi/180

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
B = Frame('B')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[1,0,0],qA,system)
B.rotate_fixed_axis_directed(A,[0,0,1],qB,system)

pNA=0*N.x

pAcm=pNA-lA*A.y
pBcm=pNA-lB*A.y

wNA = N.getw_(A)
wAB = A.getw_(B)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)
BodyB = Body('BodyB',B,pBcm,mB,IB,system)

#ParticleA = Particle(system,pAcm,mA,'ParticleA')
#ParticleB = Particle(system,pBcm,mB,'ParticleB')
#ParticleC = Particle(system,pCcm,mC,'ParticleC')

system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = BodyB.pCM.dot(N.x)
y2 = BodyB.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func1 = system.state_space_post_invert(f,ma)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,x2,y2,KE-PE,qA,qB],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(y[:,0],y[:,1])
plt.plot(y[:,2],y[:,3])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,4])

plt.figure(3)
plt.hold(True)
plt.plot(t,y[:,5:])
plt.show()
