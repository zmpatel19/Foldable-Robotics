# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
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
from math import pi
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

fx = Variable('fx')
fy = Variable('fy')

#Ixx_A = Constant('Ixx_A',8.96572844222684e-07,system)
#Iyy_A = Constant('Iyy_A',5.31645644183654e-06,system)
#Izz_A = Constant('Izz_A',5.31645644183654e-06,system)
#Ixx_B = Constant('Ixx_B',6.27600676796613e-07,system)
#Iyy_B = Constant('Iyy_B',1.98358014762822e-06,system)
#Izz_B = Constant('Izz_B',1.98358014762822e-06,system)
#Ixx_C = Constant('Ixx_C',4.39320316677997e-07,system)
#Iyy_C = Constant('Iyy_C',7.9239401855911e-07,system)
#Izz_C = Constant('Izz_C',7.9239401855911e-07,system)

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

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(B,[0,0,1],qC,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBC = pAB + lB*B.x
pCtip = pBC + lC*C.x
vCtip = pCtip.time_derivative(N,system)
aCtip = vCtip.time_derivative(N,system)

pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pBC+lC/2*C.x

wNA = N.getw_(A)
wAB = A.getw_(B)
wBC = B.getw_(C)

#IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
#IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
#IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

#BodyA = Body('BodyA',A,pAcm,mA,IA,system)
#BodyB = Body('BodyB',B,pBcm,mB,IB,system)
#BodyC = Body('BodyC',C,pCcm,mC,IC,system)

ParticleA = Particle(pAcm,mA,'ParticleA',system)
ParticleB = Particle(pBcm,mB,'ParticleB',system)
ParticleC = Particle(pCcm,mC,'ParticleC',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wBC,wBC)
system.addforce(fx*N.x,vCtip)
system.addforce(fy*N.y,vCtip)

#system.addforce(-k*(qA-preload1)*N.z,wNA)
#system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)
system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
system.add_spring_force1(k,(qB-preload2)*N.z,wAB)
system.add_spring_force1(k,(qC-preload3)*N.z,wBC)

system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)
x2 = ParticleB.pCM.dot(N.x)
y2 = ParticleB.pCM.dot(N.y)
x3 = ParticleC.pCM.dot(N.x)
y3 = ParticleC.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
#print('creating second order function...')
a=[aCtip.dot(N.x),aCtip.dot(N.y)]

import sympy
b = sympy.Matrix(a)
b.jacobian(system.get_q(2))
c=b.jacobian(system.get_q(2))
d = sympy.lambdify(system.get_state_variables(),c)

func1 = system.createsecondorderfunction4(f,ma,d)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,x2,y2,x3,y3,KE-PE,qA,qB,qC],system)
y = output.calc(states)
pynamics.toc()
#
plt.figure(1)
plt.plot(y[:,0],y[:,1])
plt.plot(y[:,2],y[:,3])
plt.plot(y[:,4],y[:,5])
plt.axis('equal')
#
plt.figure(2)
plt.plot(y[:,6])
#
plt.figure(3)
plt.plot(t,y[:,7:10])
plt.show()
