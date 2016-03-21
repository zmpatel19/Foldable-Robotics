# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

from sympy import pi
import sympy
import danamics
from danamics import *
import numpy
import scipy
import scipy.integrate
#from tictoc import *
import matplotlib.pyplot as plt

#===============================================================================
system=dynsystem()


constant('lA',.04,system)
constant('lB',.04,system)
constant('g',9.81,system)
constant('mA',.0145,system)
constant('mB',.0145,system)
constant('zero',0,system)

constant('Ixx_A',8.6e-007,system)
constant('Iyy_A',2.2e-006,system)
constant('Izz_A',2.2e-006,system)
constant('Ixx_B',8.6e-007,system)
constant('Iyy_B',2.2e-006,system)
constant('Izz_B',2.2e-006,system)

constant('b',0.00001,system)
constant('k',0.1,system)
        
#accelerationvariable('xA')
#accelerationvariable('yA')
accelerationvariable('qA',system)
accelerationvariable('xB',system)
accelerationvariable('yB',system)
accelerationvariable('qB',system)

sym_undifferentiable('fNAx')
sym_undifferentiable('fNAy')
sym_undifferentiable('fABx')
sym_undifferentiable('fABy')

initialvalues = {}
#initialvalues[xA]=.02
#initialvalues[xA_d]=0
#initialvalues[yA]=0
#initialvalues[yA_d]=0
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[xB]=.06
initialvalues[xB_d]=0
initialvalues[yB]=0
initialvalues[yB_d]=0
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180

frame('N',system)
frame('A',system)
frame('B',system)

N.setnewtonian()
A.RotateBodyZ(N,qA)
B.RotateBodyZ(A,qB)

#A.setpathtonewtonian(['A','N'])
#B.setpathtonewtonian(['B','A','N'])

pNA=zero*N.x+zero*N.y+zero*N.z

#pAcm=xA*N.x*yA*N.y
pAN = pNA
pAcm = pAN+lA/2*A.x
pBcm=xB*N.x+yB*N.y

#pAN=pAcm - lA/2*A.x
pAB=pAcm + lA/2*A.x
pBA=pBcm - lB/2*B.x

#vNA = vectorderivative(pNA,N)
#aNA = vectorderivative(vNA,N)
#
#vAN = vectorderivative(pAN,N)
#aAN = vectorderivative(vAN,N)

vAB = vectorderivative(pAB,N,system)
aAB = vectorderivative(vAB,N,system)

vBA = vectorderivative(pBA,N,system)
aBA = vectorderivative(vBA,N,system)

#constraint1 = pNA-pAN
#constraint1_d = vectorderivative(constraint1,N)
#constraint1_dd = vectorderivative(constraint1_d,N)
#
constraint2 = pAB-pBA
constraint2_d = vectorderivative(constraint2,N,system)
constraint2_dd = vectorderivative(constraint2_d,N,system)

wNA = angularvelocityN(N,A,system)
wAB = angularvelocityN(A,B,system)

body('BodyA',A,pAcm,mA,I_generic(A,Ixx_A,Iyy_A,Izz_A),system)
body('BodyB',B,pBcm,mB,I_generic(B,Ixx_B,Iyy_B,Izz_B),system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)

system.addforce(-k*qA*N.z,wNA)
system.addforce(-k*qB*A.z,wAB)

#system.addforce(fNAx*N.x+fNAy*N.y,vNA)
#system.addforce(-fNAx*N.x+-fNAy*N.y,vAN)

system.addforce(fABx*N.x+fABy*N.y,vBA)
system.addforce(-fABx*N.x+-fABy*N.y,vAB)

system.addforcegravity(-g*N.y)

x1 = dot(BodyA.pCM,N.x)
y1 = dot(BodyA.pCM,N.y)
x2 = dot(BodyB.pCM,N.x)
y2 = dot(BodyB.pCM,N.y)
KE = system.KE
PE = system.getPEGravity(pNA)
    
statevariables = system.q+system.q_d
ini = [item.subs(initialvalues) for item in statevariables]
t = scipy.arange(0,10,.01)
outputs = outputclass([x1,y1,x2,y2,KE,PE],system.constants)

#tic()
print('solving dynamics...')

junk,junk,eq = system.getdynamics()
#eq_con = [dot(constraint1_dd,N.x),dot(constraint1_dd,N.y),dot(constraint2_dd,N.x),dot(constraint2_dd,N.y)]
eq_con = [dot(constraint2_dd,B.x),dot(constraint2_dd,B.y)]
var_dd,forces = solveconstraineddynamics(list(eq.values()),eq_con,system.q_dd,[fABx,fABy])

#toc()
print('integrating...')
var_dd=var_dd.subs(system.constants)
func1 = createsecondorderfunction(var_dd,statevariables,system.q_d,func_format = 'odeint')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-8,atol=1e-8)
#toc()
print('calculating outputs..')
outputs.calc(statevariables,states)
#toc()

plt.figure(1)
plt.hold(True)
plt.plot(outputs(x1),outputs(y1))
plt.plot(outputs(x2),outputs(y2))

plt.figure(2)
plt.plot(outputs(KE)-outputs(PE))
plt.show()
#
#link1surface = surfaceclass(A,pNA,system.constants,statevariables,'../../STL/phalanx.stl',1,False)
#link2surface = surfaceclass(B,pAB,system.constants,statevariables,'../../STL/phalanx.stl',1,False)
#
#import DanGL
#DanGL.init()
#DanGL.t = t
#DanGL.surfaces = [link1surface,link2surface]
#DanGL.dt = .01
#DanGL.states = states
#DanGL.ii = 0
#DanGL.view = numpy.r_[0,0,.5,0,0,0,0,1,0]
#DanGL.run()
