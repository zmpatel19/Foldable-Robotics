# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""


from sympy import pi
import sympy
import pynamics
#pynamics.script_mode = True
from pynamics import *
import numpy
import scipy
import scipy.integrate
#from tictoc import *
import matplotlib.pyplot as plt
from pynamics.system import System
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.frame import Frame
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output

#===============================================================================
system=System()


lA = Constant(.04,'lA',system)
lB = Constant(.04,'lB',system)
g = Constant(9.81,'g',system)
mA = Constant(.0145,'mA',system)
mB = Constant(.0145,'mB',system)
zero = Constant(0,'zero',system)

Ixx_A = Constant(8.6e-007,'Ixx_A',system)
Iyy_A = Constant(2.2e-006,'Iyy_A',system)
Izz_A = Constant(2.2e-006,'Izz_A',system)
Ixx_B = Constant(8.6e-007,'Ixx_B',system)
Iyy_B = Constant(2.2e-006,'Iyy_B',system)
Izz_B = Constant(2.2e-006,'Izz_B',system)

b = Constant(0.00001,'b',system)
k = Constant(0.1,'k',system)
        
#accelerationvariable('xA')
#accelerationvariable('yA')
qA,qA_d,qA_dd = Differentiable(system,'qA')
xB,xB_d,xB_dd = Differentiable(system,'xB')
yB,yB_d,yB_dd = Differentiable(system,'yB')
qB,qB_d,qB_dd = Differentiable(system,'qB')

fNAx = Variable('fNAx')
fNAy = Variable('fNAy')
fABx = Variable('fABx')
fABy = Variable('fABy')

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

N=Frame('N')
A=Frame('A')
B=Frame('B')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
B.rotate_fixed_axis_directed(A,[0,0,1],qB,system)

#A.setpathtonewtonian(['A','N'])
#B.setpathtonewtonian(['B','A','N'])

zero = sympy.Number(0)
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

vAB = pAB.time_derivative(N,system)
aAB = vAB.time_derivative(N,system)

vBA = pBA.time_derivative(N,system)
aBA = vBA.time_derivative(N,system)

#constraint1 = pNA-pAN
#constraint1_d = vectorderivative(constraint1,N)
#constraint1_dd = vectorderivative(constraint1_d,N)
#
constraint2 = pAB-pBA
constraint2_d = constraint2.time_derivative(N,system)
constraint2_dd = constraint2_d.time_derivative(N,system)

wNA = N.getw_(A)
wAB = A.getw_(B)

IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(A,Ixx_B,Iyy_B,Izz_B)

Body('BodyA',A,pAcm,mA,IA,system)
Body('BodyB',B,pBcm,mB,IB,system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)

system.addforce(-k*qA*N.z,wNA)
system.addforce(-k*qB*A.z,wAB)

#system.addforce(fNAx*N.x+fNAy*N.y,vNA)
#system.addforce(-fNAx*N.x+-fNAy*N.y,vAN)

system.addforce(fABx*N.x+fABy*N.y,vBA)
system.addforce(-fABx*N.x+-fABy*N.y,vAB)

system.addforcegravity(-g*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = BodyB.pCM.dot(N.x)
y2 = BodyB.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA)
    
statevariables = system.get_state_variables()
ini = [item.subs(initialvalues) for item in statevariables]
t = scipy.arange(0,10,.01)
outputs = Output([x1,y1,x2,y2,KE,PE],system)

#tic()
print('solving dynamics...')

f,ma = system.getdynamics()
#eq_con = [dot(constraint1_dd,N.x),dot(constraint1_dd,N.y),dot(constraint2_dd,N.x),dot(constraint2_dd,N.y)]
print('solving constraints...')
eq_con = [constraint2_dd.dot(B.x),constraint2_dd.dot(B.y)]
var_dd,forces = system.solveconstraineddynamics(list(numpy.array(f) - numpy.array(ma)),eq_con,system.get_q(2),[fABx,fABy])

#toc()
print('creating second order function...')
var_dd 
var_dd=var_dd.subs(system.constants)
func1 = system.createsecondorderfunction_old(var_dd)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-8,atol=1e-8)
#toc()
print('calculating outputs..')
outputs.calc(statevariables,states)
#toc()

plt.figure(1)
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
