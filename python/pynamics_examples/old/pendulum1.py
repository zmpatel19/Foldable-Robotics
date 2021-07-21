# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

from math import pi
from danamics import *
import scipy
import scipy.integrate
from tictoc import *
import matplotlib.pyplot as plt

#===============================================================================

system = dynsystem()
constant('lA',.04,system)
constant('lB',.04,system)
constant('g',9.81,system)
constant('mA',.0145,system)
constant('zero',0,system)

constant('Ixx_A',8.6e-007,system)
constant('Iyy_A',2.2e-006,system)
constant('Izz_A',2.2e-006,system)

constant('b',0.001,system)
constant('k',1,system)

sympy.Symbol('fx')

accelerationvariable('qA',system)


initialvalues = {}
initialvalues[qA]=30*pi/180
initialvalues[qA_d]=0*pi/180

statevariables = system.q+system.q_d
ini = [item.subs(initialvalues) for item in statevariables]

frame('N',system)
frame('A',system)

N.setnewtonian()
A.RotateBodyZ(N,qA)

pNA=zero*N.x+zero*N.y+zero*N.z
pAB=pNA+lA*A.x
pAcm=pNA+lA/2*A.x

wNA = angularvelocityN(N,A,system)

BodyA = body('BodyA',A,pAcm,mA,I_generic(A,Ixx_A,Iyy_A,Izz_A),system)

system.addforce(-b*wNA,wNA)
system.addforce(-k*qA*N.z,wNA)
system.addforcegravity(-g*N.y)

t = scipy.arange(0,1,.0001)

tic()
print('solving dynamics...')
var_dd = system.solvedynamics('LU',False)
toc()
print('integrating...')
var_dd=var_dd.subs(system.constants)
func1 = createsecondorderfunction(var_dd,statevariables,system.q_d,func_format = 'odeint')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
toc()
print('calculating outputs..')
x1 = dot(BodyA.pCM,N.x)
y1 = dot(BodyA.pCM,N.y)
KE = system.KE
PE = system.getPEGravity(pNA)
outputs = outputclass([x1,y1,KE,PE],system.constants)
outputs.calc(statevariables,states)
toc()

plt.figure(1)
plt.plot(outputs(x1),outputs(y1))

plt.figure(2)
plt.plot(t,outputs(KE)-outputs(PE))
plt.show()
