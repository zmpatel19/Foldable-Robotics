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
import pynamics.integration

#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

tol=1e-7

l = Constant(.5,'l',system)
xO = Constant(0, 'xO',system)

M = Constant(10,'M',system)
m = Constant(10,'m',system)


I_xx = Constant(9,'I_xx',system)
I_yy = Constant(9,'I_yy',system)
I_zz = Constant(9,'I_zz',system)

g = Constant(9.81,'g',system)
b = Constant(5e1,'b',system)
k = Constant(1e3,'k',system)

tinitial = 0
tfinal = 10
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

x,x_d,x_dd = Differentiable('x',system)
q,q_d,q_dd = Differentiable('q',system)

initialvalues = {}
initialvalues[x]=.5
initialvalues[x_d]=0

initialvalues[q]=30*pi/180
initialvalues[q_d]=0*pi/180


statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],q,system)

p1 = x*N.x
p2 = p1 - l*A.y
v1 = p1.time_derivative(N,system)
v2 = p2.time_derivative(N, system)

I = Dyadic.build(A,I_xx,I_yy,I_zz)

BodyA = Body('BodyA',A,p2,m,I,system)
ParticleO = Particle(p2,M,'ParticleO',system)


stretch = x-xO
system.add_spring_force1(k,(stretch)*N.x,v1)
system.addforce(-b*v1,v1)
system.addforcegravity(-g*N.y)

f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,constants = system.constant_values)
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol,args=({'constants':{},'alpha':1e2,'beta':1e1},))

# =============================================================================
KE = system.get_KE()
PE = system.getPEGravity(0*N.x) - system.getPESprings()
energy = Output([KE-PE])
energy.calc(states,t)
energy.plot_time()
# =============================================================================
points_list = [p1,p2]
#points_list = [item2 for item in points_list for item2 in [item.dot(N.x),item.dot(N.y)]]
#points = Output(points_list)
#y = points.calc(states,t)
#y = y.reshape((-1,2,2))

#plt.figure()
#plt.plot(y[:,1,0],y[:,1,1])
#plt.axis('equal')

states2= Output([x,q])
states2.calc(states,t)

plt.figure()
plt.plot(states[:,0])
plt.figure()
plt.plot(states[:,1])

points2 = PointsOutput(points_list)
points2.calc(states,t)
#points2.plot_time()
#points2.animate(fps = 30, movie_name='cart_pendulum.mp4',lw=2)

