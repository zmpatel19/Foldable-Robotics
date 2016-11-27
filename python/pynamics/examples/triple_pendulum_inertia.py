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
lB = Constant('lB',1,system)
lC = Constant('lC',1,system)

mA = Constant('mA',1,system)
mB = Constant('mB',1,system)
mC = Constant('mC',1,system)

g = Constant('g',9.81,system)
b = Constant('b',1e1,system)
k = Constant('k',1e2,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant('preload1',0*pi/180,system)
preload2 = Constant('preload2',0*pi/180,system)
preload3 = Constant('preload3',0*pi/180,system)

Ixx_A = Constant('Ixx_A',1,system)
Iyy_A = Constant('Iyy_A',1,system)
Izz_A = Constant('Izz_A',1,system)
Ixx_B = Constant('Ixx_B',1,system)
Iyy_B = Constant('Iyy_B',1,system)
Izz_B = Constant('Izz_B',1,system)
Ixx_C = Constant('Ixx_C',1,system)
Iyy_C = Constant('Iyy_C',1,system)
Izz_C = Constant('Izz_C',1,system)

qA,qA_d,qA_dd = Differentiable(system,'qA')
qB,qB_d,qB_dd = Differentiable(system,'qB')
qC,qC_d,qC_dd = Differentiable(system,'qC')

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=0*pi/180

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
B = Frame('B')
C = Frame('C')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
B.rotate_fixed_axis_directed(A,[0,0,1],qB,system)
C.rotate_fixed_axis_directed(B,[0,0,1],qC,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBC = pAB + lB*B.x
pCtip = pBC + lC*C.x

pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pBC+lC/2*C.x

wNA = N.getw_(A)
wAB = A.getw_(B)
wBC = B.getw_(C)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)
BodyB = Body('BodyB',B,pBcm,mB,IB,system)
BodyC = Body('BodyC',C,pCcm,mC,IC,system)

#ParticleA = Particle(system,pAcm,mA,'ParticleA')
#ParticleB = Particle(system,pBcm,mB,'ParticleB')
#ParticleC = Particle(system,pCcm,mC,'ParticleC')

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wBC,wBC)

#system.addforce(-k*(qA-preload1)*N.z,wNA)
#system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)
system.add_spring_force(k,(qA-preload1)*N.z,wNA) 
system.add_spring_force(k,(qB-preload2)*N.z,wAB)
system.add_spring_force(k,(qC-preload3)*N.z,wBC)

system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = BodyB.pCM.dot(N.x)
y2 = BodyB.pCM.dot(N.y)
x3 = BodyC.pCM.dot(N.x)
y3 = BodyC.pCM.dot(N.y)
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
output = Output([x1,y1,x2,y2,x3,y3,KE-PE,qA,qB,qC],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(y[:,0],y[:,1])
plt.plot(y[:,2],y[:,3])
plt.plot(y[:,4],y[:,5])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,6])

plt.figure(3)
plt.hold(True)
plt.plot(t,y[:,7:10])
plt.show()
