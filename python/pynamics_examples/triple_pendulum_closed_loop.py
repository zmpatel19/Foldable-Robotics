# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
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

lA = Constant(1,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(1,'lC',system)
lD = Constant(1,'lD',system)

mA = Constant(1,'mA',system)
mB = Constant(.5,'mB',system)
mC = Constant(1,'mC',system)
mD = Constant(.5,'mD',system)

g = Constant(9.81,'g',system)
b = Constant(1e1,'b',system)
k = Constant(1e2,'k',system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)
preload2 = Constant(0*pi/180,'preload2',system)
preload3 = Constant(0*pi/180,'preload3',system)
preload4 = Constant(0*pi/180,'preload4',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

initialvalues = {}
initialvalues[qA]=1*pi/180
initialvalues[qA_d]=10*pi/180
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=0*pi/180
initialvalues[qD]=0*pi/180
initialvalues[qD_d]=0*pi/180

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
A = Frame('A')
B = Frame('B')
C = Frame('C')
D = Frame('D')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
B.rotate_fixed_axis_directed(A,[0,0,1],qB,system)
C.rotate_fixed_axis_directed(N,[0,0,1],qC,system)
D.rotate_fixed_axis_directed(C,[0,0,1],qD,system)

pNA=0*N.x
pAB=pNA-lA*A.y
pBtip = pAB + lB*B.x

pNC = pNA+1*N.x
pCD = pNC - lC*C.y
#pDtip = pNC + lC*C.x

pAcm=pNA-lA/2*A.y
pBcm=pAB+lB/2*B.x
pCcm=pNC-lC/2*C.y
pDcm=pCD-lD/2*D.x

wNA = N.getw_(A)
wAB = A.getw_(B)
#wBC = B.getw_(C)

#IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
#IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
#IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

#BodyA = Body('BodyA',A,pAcm,mA,IA,system)
#BodyB = Body('BodyB',B,pBcm,mB,IB,system)
#BodyC = Body('BodyC',C,pCcm,mC,IC,system)

ParticleA = Particle(pAcm,mA,'ParticleA',system)
ParticleB = Particle(pBcm,mB,'ParticleB',system)
ParticleC = Particle(pCcm,mC,'ParticleC',system)
ParticleD = Particle(pDcm,mD,'ParticleD',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
#system.addforce(-b*wBC,wBC)

system.addforce(-k*(qA-preload1)*N.z,wNA)
system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)
#system.add_spring_force(k,(qA-preload1)*N.z,wNA) 
#system.add_spring_force(k,(qB-preload2)*N.z,wAB)
#system.add_spring_force(k,(qC-preload3)*N.z,wBC)

system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)
x2 = ParticleB.pCM.dot(N.x)
y2 = ParticleB.pCM.dot(N.y)
x3 = ParticleC.pCM.dot(N.x)
y3 = ParticleC.pCM.dot(N.y)
x4 = ParticleD.pCM.dot(N.x)
y4 = ParticleD.pCM.dot(N.y)

eq1 = B.x.dot(D.x)
#eq1_d = system.derivative(eq1)
#eq1_dd = system.derivative(eq1_d)
#eq1_d = eq1.diff_in_parts(N,system)
#eq1_dd = eq1_d.diff_in_parts(N,system)

eq2 = x4 - x2
eq3 = y4 - y2
#eq2 = ParticleB.aCM

eq = [eq1,eq2,eq3]
eq_d= [system.derivative(item) for item in eq]
eq_dd= [system.derivative(item) for item in eq_d]

KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()
    
pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func1 = system.state_space_post_invert(f,ma,eq_dd)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14, args=({'constants':system.constant_values},))
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,x2,y2,x3,y3,KE-PE,qA,qB,qC],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.plot(y[:,0],y[:,1])
plt.plot(y[:,2],y[:,3])
plt.plot(y[:,4],y[:,5])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,6])

plt.figure(3)
plt.plot(t,y[:,7:10])
plt.show()
