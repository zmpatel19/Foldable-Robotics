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

import sympy
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
import math
system = System()



lA = Constant('lA',.04,system)
lB = Constant('lB',.04,system)
g = Constant('g',9.81,system)
mA = Constant('mA',.0145,system)
#Constant('zero',0,system)

Ixx_A = Constant('Ixx_A',8.6e-007,system)
Iyy_A = Constant('Iyy_A',2.2e-006,system)
Izz_A = Constant('Izz_A',2.2e-006,system)

b = Constant('b',0.00001,system)
k = Constant('k',0.0,system)
fx = sympy.Symbol('fx')
qA,qA_d,qA_dd = Differentiable(system,'qA')

initialvalues = {}
initialvalues[qA]=30*math.pi/180
initialvalues[qA_d]=0*math.pi/180

statevariables = [qA,qA_d]
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pAcm=pNA+lA/2*A.x

wNA = N.getw_(A)

BodyA = Body('BodyA',A,pAcm,mA,Dyadic.build(A,Ixx_A,Iyy_A,Izz_A),system)

system.addforce(-b*wNA,wNA)
system.addforce(-k*qA*N.z,wNA)
system.addforcegravity(-g*N.y)

t = scipy.arange(0,10,.01)
pynamics.tic()
print('solving dynamics...')
var_dd = system.solvedynamics('LU',False)
pynamics.toc()
print('integrating...')
var_dd=var_dd.subs(system.constants)
func1 = system.createsecondorderfunction(var_dd,statevariables,system.get_q(1),func_format = 'odeint')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-8,atol=1e-8)
pynamics.toc()
print('calculating outputs..')
x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA)
outputs = Output([x1,y1,KE,PE],system.constants)
outputs.calc(statevariables,states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(outputs(x1),outputs(y1))

plt.figure(2)
plt.plot(t,outputs(KE)-outputs(PE))
plt.show()
