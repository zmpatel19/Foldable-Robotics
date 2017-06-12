# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output

import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()

Constant('lA',0.0570776,system)
Constant('lB',0.0399543,system)
Constant('lC',0.027968,system)
Constant('g',9.81,system)
Constant('mA',0.0179314568844537,system)
Constant('mB',0.0125520135359323,system)
#Constant('mC',0.00878640633355993,system)
#Constant('zero',0,system)

Constant('Ixx_A',8.96572844222684e-07,system)
Constant('Iyy_A',5.31645644183654e-06,system)
Constant('Izz_A',5.31645644183654e-06,system)
Constant('Ixx_B',6.27600676796613e-07,system)
Constant('Iyy_B',1.98358014762822e-06,system)
Constant('Izz_B',1.98358014762822e-06,system)
#Constant('Ixx_C',4.39320316677997e-07,system)
#Constant('Iyy_C',7.9239401855911e-07,system)
#Constant('Izz_C',7.9239401855911e-07,system)

Constant('b',1e-4,system)
Constant('k',1.0,system)
Constant('preload1',-90*pi/180,system)
Constant('preload2',0*pi/180,system)
#Constant('preload3',0*pi/180,system)
        
Differentiable(system,'qA')
Differentiable(system,'qB')
#Differentiable('qC',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
#initialvalues[qC]=0*pi/180
#initialvalues[qC_d]=0*pi/180

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

Frame('N')
Frame('A')
Frame('B')
#Frame('C')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
B.rotate_fixed_axis_directed(A,[0,0,1],qB,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBC = pAB + lB*B.x
#pCtip = pBC + lC*C.x

pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
#pCcm=pBC+lC/2*C.x

wNA = N.getw_(A)
wAB = A.getw_(B)
#wBC = B.getw_(C)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
#IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

Body('BodyA',A,pAcm,mA,IA,system)
Body('BodyB',B,pBcm,mB,IB,system)
#Body('BodyC',C,pCcm,mC,IC,system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
#system.addforce(-b*wBC,wBC)

system.addforce(-k*(qA-preload1)*N.z,wNA)
system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)

system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = BodyB.pCM.dot(N.x)
y2 = BodyB.pCM.dot(N.y)
#x3 = BodyC.pCM.dot(N.x)
#y3 = BodyC.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA)-.5*k*(qA-preload1)**2-.5*k*(qB-preload2)**2
    
t = numpy.r_[0:10:.001]
outputs = Output([x1,y1,x2,y2,KE,PE,qA,qB],system.constants)

pynamics.tic()
print('solving dynamics...')
var_dd = system.solvedynamics('LU',True)
pynamics.toc()
print('integrating...')
var_dd=var_dd.subs(system.constants)
func1 = system.createsecondorderfunction(var_dd,statevariables,system.get_q(1),func_format = 'odeint')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
outputs.calc(statevariables,states)
pynamics.toc()

plt.figure(1)
plt.plot(outputs(x1),outputs(y1))
plt.plot(outputs(x2),outputs(y2))
#plt.plot(outputs(x3),outputs(y3))
plt.axis('equal')

plt.figure(2)
plt.plot(outputs(KE)-outputs(PE))

plt.figure(3)
plt.plot(t,outputs(qA))
plt.plot(t,outputs(qB))
#plt.plot(t,outputs(qC))

plt.show()
