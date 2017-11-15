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
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
pynamics.script_mode = True

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()


g = Constant(9.81,'g',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

Differentiable('qA')
Differentiable('qB')
Differentiable('qC')

Differentiable('x')
Differentiable('y')
Differentiable('z')

Constant(1,'mC')
Constant(2,'Ixx')
Constant(3,'Iyy')
Constant(1,'Izz')

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=7
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=.2
initialvalues[qC]=0*pi/180
initialvalues[qC_d]=.2

initialvalues[x]=0
initialvalues[x_d]=10
initialvalues[y]=0
initialvalues[y_d]=10
initialvalues[z]=0
initialvalues[z_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

Frame('N')
Frame('A')
Frame('B')
Frame('C')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[1,0,0],qA,system)
B.rotate_fixed_axis_directed(A,[0,1,0],qB,system)
C.rotate_fixed_axis_directed(B,[0,0,1],qC,system)


pCcm=x*N.x+y*N.y+z*N.z
wNC = N.getw_(C)

#IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
#IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx,Iyy,Izz)

#BodyA = Body('BodyA',A,pAcm,mA,IA,system)
#BodyB = Body('BodyB',B,pBcm,mB,IB,system)
Body('BodyC',C,pCcm,mC,IC)
#
#ParticleB = Particle(pBcm,mB,'ParticleB',system)
#ParticleC = Particle(pCcm,mC,'ParticleC',system)
#
#system.addforce(-b*wNA,wNA)
#system.addforce(-b*wAB,wAB)
#system.addforce(-b*wBC,wBC)
#
##system.addforce(-k*(qA-preload1)*N.z,wNA)
##system.addforce(-k*(qB-preload2)*A.z,wAB)
##system.addforce(-k*(qC-preload3)*B.z,wBC)
#system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
#system.add_spring_force1(k,(qB-preload2)*N.z,wAB)
#system.add_spring_force1(k,(qC-preload3)*N.z,wBC)
#
system.addforcegravity(-g*N.y)
#


points = [0*N.x,pCcm]

ang = [wNC.dot(C.x),wNC.dot(C.y),wNC.dot(C.z)]
#ang = [wNC.dot(N.x),wNC.dot(N.y),wNC.dot(N.z)]

#x1 = ParticleA.pCM.dot(N.x)
#y1 = ParticleA.pCM.dot(N.y)
#x2 = ParticleB.pCM.dot(N.x)
#y2 = ParticleB.pCM.dot(N.y)
#x3 = ParticleC.pCM.dot(N.x)
#y3 = ParticleC.pCM.dot(N.y)
#pynamics.tic()
#print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')
func1 = system.state_space_post_invert(f,ma)
#print('integrating...')
states=scipy.integrate.odeint(func1,ini,t, args=({'constants':system.constant_values},))
#pynamics.toc()
#print('calculating outputs..')
#
#KE = system.get_KE()
#PE = system.getPEGravity(pNA) - system.getPESprings()
#    
output = Output(ang,system)
output.calc(states)
output.plot_time()


po = PointsOutput(points,system)
po.calc(states)
po.animate()
#pynamics.toc()
#
#plt.figure(1)
#plt.plot(y[:,0],y[:,1])
#plt.plot(y[:,2],y[:,3])
#plt.plot(y[:,4],y[:,5])
#plt.axis('equal')
#
#plt.figure(2)
#plt.plot(y[:,6])
#
#plt.figure(3)
#plt.plot(t,y[:,7:10])
#plt.show()
