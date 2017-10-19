# In The Name Of GOD
#"""
#Written by Daniel M. Aukes
#Email: danaukes<at>gmail.com
#Please see LICENSE for full license.
#"""


import os
os.system('cls')  # on windows
import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle
import math

import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.mplot3d import Axes3D

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
#from sympy import pi


pynamics.tic()

system = System()
import sympy
from sympy import atan2
from math import pi,sin,cos


tinitial = 0
tfinal = 10
tstep = 0.05
t = numpy.r_[tinitial:tfinal:tstep]



av = 0.05
bv = 0.20

lAA = 0.08
lBB = 0.05
lCC = bv - lAA - lBB


lA = Constant(lAA,'lA',system)
lB = Constant(lBB,'lB',system)
lC = Constant(lCC,'lC',system)

mA = Constant(0.2,'mA',system)
mB = Constant(0.2,'mB',system)
mC = Constant(0.2,'mC',system)

g = Constant(9.81,'g',system)
b = Constant(0.51e-5,'b',system)
#k = Constant(0.0114,'k',system)

k = Constant(1e1,'k',system)


preload1 = Constant(0*pi/180,'preload1',system)
preload2 = Constant(0*pi/180,'preload2',system)
preload3 = Constant(0*pi/180,'preload3',system)

Ixx_A = Constant((av * pow(lA,3))/12,'Ixx_A',system)
Iyy_A = Constant((lA * pow(av,3))/12,'Iyy_A',system)
Izz_A = Constant(1,'Izz_A',system)
Ixx_B = Constant((av * pow(lB,3))/12,'Ixx_B',system)
Iyy_B = Constant((lB * pow(av,3))/12,'Iyy_B',system)
Izz_B = Constant(1,'Izz_B',system)
Ixx_C = Constant((av * pow(lC,3))/12,'Ixx_C',system)
Iyy_C = Constant((lC * pow(av,3))/12,'Iyy_C',system)
Izz_C = Constant(1,'Izz_C',system)

#Ixx_A = Constant(1,'Ixx_A',system)
#Iyy_A = Constant(1,'Iyy_A',system)
#Izz_A = Constant(1,'Izz_A',system)
#Ixx_B = Constant(1,'Ixx_B',system)
#Iyy_B = Constant(1,'Iyy_B',system)
#Izz_B = Constant(1,'Izz_B',system)
#Ixx_C = Constant(1,'Ixx_C',system)
#Iyy_C = Constant(1,'Iyy_C',system)
#Izz_C = Constant(1,'Izz_C',system)


qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
Hx,Hx_d,Hx_dd = Differentiable('Hx',system)
Hy,Hy_d,Hy_dd = Differentiable('Hy',system)





initialvalues = {}
initialvalues[qA]=90*pi/180
initialvalues[qA_d]=0.01*pi/180
initialvalues[qB]=90*pi/180
initialvalues[qB_d]=0.01*pi/180
initialvalues[qC]=90*pi/180
initialvalues[qC_d]=0.01*pi/180
initialvalues[Hx]=0.01
initialvalues[Hx_d]=0
initialvalues[Hy]=0.01
initialvalues[Hy_d]=0

#initialvalues[alphaA]=0*pi/180
#initialvalues[alphaB]=0*pi/180
#initialvalues[alphaC]=0*pi/180


#rho = Constant(1000,'rho',system)
#SA = Constant(av*lA,'SA',system)
#SB = Constant(av*lB,'SB',system)
#SC = Constant(av*lC,'SC',system)

rho = 1000
SA = av*lA
SB = av*lB
SC = av*lC


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



pNA = Hx*N.x + Hy*N.y
#pNA = 0*N.x
pAB = pNA + lA * A.x
pBC = pAB + lB * B.x
pCtip = pBC + lC *C .x

pAcm= pNA + lA/2 * A.x
pBcm= pAB + lB/2 * B.x
pCcm= pBC + lC/2 * C.x

vAcm = pAcm.time_derivative(N,system)
vBcm = pBcm.time_derivative(N,system)
vCcm = pCcm.time_derivative(N,system)

wNA = N.getw_(A)
wAB = A.getw_(B)
wBC = B.getw_(C)


IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

BodyA = Body('BodyA',A,pAcm,mA,IA,system)
BodyB = Body('BodyB',B,pBcm,mB,IB,system)
BodyC = Body('BodyC',C,pCcm,mC,IC,system)

#ParticleA = Particle(pAcm,mA,'ParticleA',system)
#ParticleB = Particle(pBcm,mB,'ParticleB',system)
#ParticleC = Particle(pCcm,mC,'ParticleC',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wAB,wAB)
system.addforce(-b*wBC,wBC)



system.addforce(0*A.y,vAcm)

#system.addforce(-k*(qA-preload1)*N.z,wNA)
#system.addforce(-k*(qB-preload2)*A.z,wAB)
#system.addforce(-k*(qC-preload3)*B.z,wBC)

system.add_spring_force(k,(qA-preload1)*N.z,wNA) 
system.add_spring_force(k,(qB-preload2)*N.z,wAB)
system.add_spring_force(k,(qC-preload3)*N.z,wBC)

system.addforcegravity(-0*N.y)

x1 = BodyA.pCM.dot(N.x)
y1 = BodyA.pCM.dot(N.y)
x2 = BodyB.pCM.dot(N.x)
y2 = BodyB.pCM.dot(N.y)
x3 = BodyC.pCM.dot(N.x)
y3 = BodyC.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

vx1= vAcm.dot(N.x)
vy1= vAcm.dot(N.y)
vx2= vBcm.dot(N.x)
vy2= vBcm.dot(N.y)
vx3= vCcm.dot(N.x)
vy3= vCcm.dot(N.y)

VP1 = vAcm.cross(pAcm)
NVP1 = (VP1.dot(VP1)) ** 0.5
VP2 = vBcm.cross(pBcm)
NVP2 = (VP2.dot(VP2)) ** 0.5
VP3 = vCcm.cross(pCcm)
NVP3 = (VP3.dot(VP3)) ** 0.5

Nv1 = (vAcm.dot(vAcm))**0.5
Nv2 = (vBcm.dot(vBcm))**0.5
Nv3 = (vCcm.dot(vCcm))**0.5

Np1 = (pAcm.dot(pAcm))**0.5
Np2 = (pBcm.dot(pBcm))**0.5
Np3 = (pCcm.dot(pCcm))**0.5

alpha1 = atan2(NVP1,vAcm.dot(pAcm))
alpha2 = atan2(NVP1,vBcm.dot(pBcm))
alpha3 = atan2(NVP1,vCcm.dot(pCcm))

#alpha1 = atan2(vx1*y1-vy1*x1,x1*vx1+y1*vy1)
#alpha2 = atan2(vx1*y1-vy1*x1,x1*vx1+x2*vx2)
#alpha3 = atan2(vx1*y1-vy1*x1,x1*vx1+x2*vx2)

FD1 = rho * pow(Nv1,2-1) * pow(sympy.sin(alpha1),2) * SA * (-vAcm)
FD2 = rho * pow(Nv2,2-1) * pow(sympy.sin(alpha2),2) * SB * (-vBcm)
FD3 = rho * pow(Nv3,2-1) * pow(sympy.sin(alpha3),2) * SC * (-vCcm)

#FD1 = rho * sympy.sin(alpha1) * (vAcm)
#FD2 = rho * sympy.sin(alpha2) * (vBcm)
#FD3 = rho * sympy.sin(alpha3) * (vCcm)

DV1 = 1/Nv1 * vAcm
DV2 = 1/Nv2 * vBcm
DV3 = 1/Nv3 * vCcm

DL1 = DV1.cross(N.z)
DL2 = DV2.cross(N.z)
DL3 = DV3.cross(N.z)


FL1 = rho * pow(Nv1,2) * sympy.sin(alpha1) * sympy.cos(alpha1) * SA * (DL1)
FL2 = rho * pow(Nv2,2) * sympy.sin(alpha2) * sympy.cos(alpha2) * SB * (DL2)
FL3 = rho * pow(Nv3,2) * sympy.sin(alpha3) * sympy.cos(alpha3) * SC * (DL3)

system.addforce(FD1,vAcm)
system.addforce(FD2,vBcm)
system.addforce(FD3,vCcm)


system.addforce(FL1,vAcm)
system.addforce(FL2,vBcm)
system.addforce(FL3,vCcm)

#cd  = 0.2 * sympy.sin(2 * pi * system.t);

#Input_Torque = 0.5 * sympy.sin(2 * pi * system.t);
#
#w1 = N.getw_(A)
#
#system.addforce(Input_Torque*N.z,w1)

    
print('solving dynamics...')
f,ma = system.getdynamics()
pynamics.toc()

print('creating second order function...')
func1 = system.state_space_post_invert(f,ma)
pynamics.toc()
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-3,atol=1e-3)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,x2,y2,x3,y3,KE-PE,qA,qB,qC,Hx,Hy,FL1,FL2,FL3,FD1,FD2,FD3],system)
y = output.calc(states)
pynamics.toc()

print('Hinge angles')
plt.figure(1)
plt.plot(t,180/pi * y[:,7:10])
plt.show()

print('x position of the head')
plt.figure(2)
plt.plot(t,y[:,10])
plt.show()

print('y position of the head')
plt.figure(3)
plt.plot(t,y[:,11])
plt.show()



print('Energy of the system')
plt.figure(4)
plt.plot(t,y[:,6])
plt.show()

print('The overall time:')
pynamics.toc()

#o2 = [pNA,pAB,pBC,pCtip]
#o2 = [item2 for item in o2 for item2 in [item.dot(N.x),item.dot(N.y),item.dot(N.z)]]
#o2 = Output(o2,system)
#y2 = o2.calc(states)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#import idealab_tools.matplotlib_tools as mm
#
#import idealab_tools.makemovie
#idealab_tools.makemovie.prep_folder()
#
#y2 = y2.reshape((-1,4,3))
#jj = 0
#for item in y2:
#    ax.cla()
#    ax.plot(item[:,0],item[:,1])
##    plt.axis('equal')
##    mm.equal_axes(ax,y2.reshape((-1,3),order=0))
#    plt.savefig('render/{0:04d}.png'.format(jj))
#    jj+=10
#
#idealab_tools.makemovie.render(image_name_format='%04d.png')
#idealab_tools.makemovie.clear_folder(rmdir=True)
