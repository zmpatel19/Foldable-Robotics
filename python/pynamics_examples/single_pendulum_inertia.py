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

lA = Constant(.075,'lA',system)

mA = Constant(.01,'mA',system)

g = Constant(9.81,'g',system)

tinitial = 0
tfinal = 4
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)

Ixx_A = Constant(50/1000/100/100,'Ixx_A',system)
Iyy_A = Constant(50/1000/100/100,'Iyy_A',system)
Izz_A = Constant(50/1000/100/100,'Izz_A',system)

qA,qA_d,qA_dd = Differentiable(system,'qA')

initialvalues = {}
initialvalues[qA]=90*pi/180
initialvalues[qA_d]=0*pi/180

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)

pNA=0*N.x
#pAB=pNA+lA*A.x

pAcm=pNA-lA*A.y

wNA = N.getw_(A)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)

#BodyA = Particle(pAcm,mA,'ParticleA',system)
#ParticleB = Particle(pBcm,mB,'ParticleB',system)
#ParticleC = Particle(pCcm,mC,'ParticleC',system)


#system.addforce(-k*(qA-preload1)*N.z,wNA)
#system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)

system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
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
plt.plot(t,y[:,3:]*180/pi)
plt.show()
