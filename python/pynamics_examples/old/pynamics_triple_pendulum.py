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

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

lA = Constant('lA',0.0570776,system)
lB = Constant('lB',0.0399543,system)
lC = Constant('lC',0.027968,system)
g = Constant('g',9.81,system)
mA = Constant('mA',0.0179314568844537,system)
mB = Constant('mB',0.0125520135359323,system)
mC = Constant('mC',0.00878640633355993,system)

Ixx_A = Constant('Ixx_A',8.96572844222684e-07,system)
Iyy_A = Constant('Iyy_A',5.31645644183654e-06,system)
Izz_A = Constant('Izz_A',5.31645644183654e-06,system)
Ixx_B = Constant('Ixx_B',6.27600676796613e-07,system)
Iyy_B = Constant('Iyy_B',1.98358014762822e-06,system)
Izz_B = Constant('Izz_B',1.98358014762822e-06,system)
Ixx_C = Constant('Ixx_C',4.39320316677997e-07,system)
Iyy_C = Constant('Iyy_C',7.9239401855911e-07,system)
Izz_C = Constant('Izz_C',7.9239401855911e-07,system)

b = Constant('b',1e-4,system)
k = Constant('k',0.0,system)
preload1 = Constant('preload1',-90*pi/180,system)
preload2 = Constant('preload2',0*pi/180,system)
preload3 = Constant('preload3',0*pi/180,system)
        
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
    
t = numpy.r_[0:5:.001]
outputs = Output([x1,y1,x2,y2,x3,y3,KE,PE,qA,qB,qC],system.constants)

pynamics.tic()
print('solving dynamics...')
var_dd = system.solvedynamics('LU',auto_z=True)
pynamics.toc()
print('substituting constants...')
var_dd=var_dd.subs(system.constants)
print('creating second order function...')
func1 = system.createsecondorderfunction(var_dd,statevariables,system.get_q(1),func_format = 'odeint')
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
outputs.calc(statevariables,states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(outputs(x1),outputs(y1))
plt.plot(outputs(x2),outputs(y2))
plt.plot(outputs(x3),outputs(y3))
plt.axis('equal')

plt.figure(2)
plt.plot(outputs(KE)-outputs(PE))

plt.figure(3)
plt.hold(True)
plt.plot(t,outputs(qA))
plt.plot(t,outputs(qB))
plt.plot(t,outputs(qC))

plt.show()
