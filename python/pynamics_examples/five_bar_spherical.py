# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.script_mode = False

from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output, PointsOutput3D
from pynamics.particle import Particle
import pynamics.integration
from pynamics.constraint import AccelerationConstraint,KinematicConstraint

import matplotlib.pyplot as plt
# plt.ion()


#import sympy
import numpy
#import matplotlib.pyplot as plt
#plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)
tol = 1e-10
from math import pi,sin,cos

#lA = Constant('lA',1,system)
#lB = Constant('lB',1,system)
#lC = Constant('lC',1,system)

################################################
#This is where we set the link angles

t1 = 50*pi/180
t2 = 70*pi/180
t3 = 70*pi/180
t4 = 60*pi/180
t0 = 2*pi-(t1+t2+t3+t4)

################################################
#mass of the system
m = Constant(1,'m',system)

################################################
#Define constant data types, seeded with constant values
t0 = Constant(t0,'t0',system)
t1 = Constant(t1,'t1',system)
t2 = Constant(t2,'t2',system)
t3 = Constant(t3,'t3',system)
t4 = Constant(t4,'t4',system)

################################################
#other constants
g = Constant(0,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e2,'k',system)

################################################
#time parameters
tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

################################################
#State variables
qA1,qA1_d,qA1_dd = Differentiable('qA1',system)
qA2,qA2_d,qA2_dd = Differentiable('qA2',system)
qA3,qA3_d,qA3_dd = Differentiable('qA3',system)
qB1,qB1_d,qB1_dd = Differentiable('qB1',system)
qB2,qB2_d,qB2_dd = Differentiable('qB2',system)

################################################
#Define initial values for state variables
initialvalues = {}
initialvalues[qA1]=1*pi/180
initialvalues[qA2]=1*pi/180
initialvalues[qA3]=1*pi/180
initialvalues[qB1]=-1*pi/180
initialvalues[qB2]=-1*pi/180

initialvalues[qA1_d]=0*pi/180
initialvalues[qA2_d]=0*pi/180
initialvalues[qA3_d]=0*pi/180
initialvalues[qB1_d]=0*pi/180
initialvalues[qB2_d]=0*pi/180

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

################################################
#Create Frames
N = Frame('N')
A1 = Frame('A1')
A12 = Frame('A12')
A2 = Frame('A2')
A23 = Frame('A23')
A3 = Frame('A3')
A34 = Frame('A34')

NB1 = Frame('NB1')
B1 = Frame('B1')
B12 = Frame('B12')
B2 = Frame('B2')
B23 = Frame('B23')

################################################
#Relative frame rotations from newtonian out to distal frames
system.set_newtonian(N)

A1.rotate_fixed_axis_directed(N,[1,0,0],qA1,system)
A12.rotate_fixed_axis_directed(A1,[0,0,1],t1,system)
A2.rotate_fixed_axis_directed(A12,[1,0,0],qA2,system)
A23.rotate_fixed_axis_directed(A2,[0,0,1],t2,system)
A3.rotate_fixed_axis_directed(A23,[1,0,0],qA3,system)
# A34.rotate_fixed_axis_directed(A3,[0,0,1],t3,system)
#
NB1.rotate_fixed_axis_directed(N,[0,0,1],-t0,system)
B1.rotate_fixed_axis_directed(NB1,[1,0,0],qB1,system)
B12.rotate_fixed_axis_directed(B1,[0,0,1],-t4,system)
B2.rotate_fixed_axis_directed(B12,[1,0,0],qB2,system)
B23.rotate_fixed_axis_directed(B2,[0,0,1],-t3,system)


################################################
#Define particles at the center of mass of each body
pNO = 0*N.x

ParticleA1 = Particle(A1.x+A12.x,m,'ParticleA1',system)
ParticleA2 = Particle(A2.x+A23.x,m,'ParticleA2',system)
# ParticleA3 = Particle(A3.x+A34.x,m/2,'ParticleA3',system)
ParticleB1 = Particle(B1.x+B12.x,m,'ParticleB1',system)
ParticleB2 = Particle(B2.x+B23.x,m,'ParticleB2',system)

################################################
#Get the relative rotational velocity between frames
wA1 = N.getw_(A1)
wA2 = A12.getw_(A2)
wA3 = A23.getw_(A3)
wB1 = NB1.getw_(B1)
wB2 = B12.getw_(B2)

################################################
#Add damping between joints
system.addforce(-b*wA1,wA1)
system.addforce(-b*wA2,wA2)
system.addforce(-b*wA3,wA3)
system.addforce(-b*wB1,wB1)
system.addforce(-b*wB2,wB2)

#system.addforce(1*A1.x,wA1)

################################################
#Add spring forces to two joints
system.add_spring_force1(k,(qA1-pi/180*45)*A1.x,wA1) 
system.add_spring_force1(k,(qA2)*A2.x,wA2) 
system.add_spring_force1(k,(qA3)*A3.x,wA3) 
system.add_spring_force1(k,(qB1+pi/180*45)*B1.x,wB1) 
system.add_spring_force1(k,(qB2)*B2.x,wB2) 

################################################
#Add gravity
system.addforcegravity(-g*N.z)


################################################
#variables for constraint equation solving
v1 = B23.x - A3.x
v2 = B23.y - A3.y
v3 = B23.z - A3.z

eq1 = []

# eq1.append(v1)
# eq1.append(v2)
# eq1.append(v3)
# #eq1.append(B23.x.dot(A3.x)-1)
# #eq1.append(A34.x.dot(B2.x)-1)
# #eq1.append((ParticleA3.pCM-ParticleB2.pCM).dot(N.y))
# #eq1.append((ParticleA3.pCM-ParticleB2.pCM).dot(N.x))
# #eq1.append((ParticleA3.pCM-ParticleB2.pCM).dot(N.z))
# #eq1 = []
# #eq1.append((B23.x-A3.x).dot(N.x))
# #eq1.append((B23.x-A3.x).dot(N.z))
# #eq1.append((A34.x-B2.x).dot(N.x))
# #eq1.append((A34.x-B2.x).dot(N.z))
# #eq1.append((ParticleA3.pCM-ParticleB2.pCM).dot(N.z))

# #take the derivative of those equations twice
# eq1_d=[(item.time_derivative()) for item in eq1]
# eq1_dd=[(item.time_derivative()) for item in eq1_d]

# eq_scalar= []
# eq_scalar.append(eq1_dd[0].length())
# eq_scalar.append(eq1_dd[1].length())
# eq_scalar.append(eq1_dd[2].length())

eq1.append(v1.dot(v1))
eq1.append(v2.dot(v2))
eq1.append(v3.dot(v3))

#take the derivative of those equations twice
eq1_d=[(system.derivative(item)) for item in eq1]
eq_scalar=[(system.derivative(item)) for item in eq1_d]

system.add_constraint(AccelerationConstraint(eq_scalar))

#x1 = ParticleA.pCM.dot(N.x)
#y1 = ParticleA.pCM.dot(N.y)
#x2 = ParticleB.pCM.dot(N.x)
#y2 = ParticleB.pCM.dot(N.y)
#x3 = ParticleC.pCM.dot(N.x)
#y3 = ParticleC.pCM.dot(N.y)
################################################
#retrive the energy of the system for plotting purposes

    
################################################
#solve equations
################################################
#This is F and MA
f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,constants=system.constant_values)
################################################
#this is the function you integrate
# func1 = system.state_space_post_invert2(f,ma,eq1_dd,eq1_d,eq1)

#states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=1e-5,atol=1e-5)
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol)

KE = system.get_KE()
PE = system.getPEGravity(pNO) - system.getPESprings()

output = Output([KE-PE],system)
y = output.calc(states,t)

plt.figure()
plt.plot(y[:])
plt.show()

o2 = [pNO,A1.x,A2.x,pNO,A2.x,A3.x,pNO,B1.x,B2.x,pNO,B2.x,B23.x,pNO]
points_output = PointsOutput3D(o2,system)
y = points_output.calc(states,t)
# points_output.plot_time()
#points_output.animate(fps = 30,movie_name = 'render.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-')

# self  = points_output
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# f=plt.figure()
# ax = f.add_subplot(1,1,1,autoscale_on=False,projection='3d')
# stepsize = 1
# ax.plot3D(xs=self.y[::stepsize,1,0],ys=self.y[::stepsize,1,1],zs=self.y[::stepsize,1,2])
# # ax.axis('equal')