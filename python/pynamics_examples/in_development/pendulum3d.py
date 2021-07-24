# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle
import pynamics.integration

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sympy
import numpy
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

mp1 = Constant(1,'mp1')
g = Constant(9.81,'g')
l1 = Constant(1,'l1')
b = Constant(1,'b')
k = Constant(1e1,'k',system)

q1,q1_d,q1_dd = Differentiable('q1',ini = [0,1])
q2,q2_d,q2_dd = Differentiable('q2',ini = [0,0])
q3,q3_d,q3_dd = Differentiable('q3',ini = [0,0])

f1 = Frame()
f2 = Frame()
f3 = Frame()
f4 = Frame()
#f5 = Frame()

system.set_newtonian(f1)
f2.rotate_fixed_axis(f1,[0,0,1],q1,system)
f3.rotate_fixed_axis(f2,[1,0,0],q2,system)
f4.rotate_fixed_axis(f3,[0,1,0],q3,system)

p0 = 0*f1.x
p1 = p0-l1*f4.x
v1=p1.time_derivative(f1)

wNA = f1.getw_(f2)


particle1 = Particle(p1,mp1)
body1 = Body('body1',f4,p1,mp1,Dyadic.build(f4,1,1,1),system = None)

#system.addforce(-b*v1,v1)
system.addforcegravity(-g*f1.z)
#system.add_spring_force1(k,(q1)*f1.z,wNA) 

points = [particle1.pCM]

points_x = [item.dot(f1.x) for item in points]
points_y = [item.dot(f1.y) for item in points]
points_z = [item.dot(f1.z) for item in points]

output_x = Output(points_x)
output_y = Output(points_y)
output_z = Output(points_z)

f,ma = system.getdynamics()
func = system.state_space_post_invert(f,ma)
t = numpy.r_[0:5:.001]
states=pynamics.integration.integrate_odeint(func,system.get_ini(),t,atol=1e-5,rtol = 1e-5, args=({'constants':system.constant_values},))
x = output_x.calc(states)
y = output_y.calc(states)
z = output_z.calc(states)

KE = system.get_KE()
PE = system.getPEGravity(0*f1.x) - system.getPESprings()

output = Output([KE-PE])
outputs = output.calc(states)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.plot(x,y,z)
plt.axis('equal')

plt.figure()
plt.plot(outputs[:])
