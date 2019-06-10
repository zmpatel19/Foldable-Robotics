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
#from pynamics.particle import Particle
import pynamics.integration
pynamics.script_mode = True
import sympy
#import logging
#pynamics.logger.setLevel(logging.ERROR)
#pynamics.system.logger.setLevel(logging.ERROR)

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi, sin, cos
system = System()


tinitial = 0
tfinal = 2.5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

ang_ini = 0

Differentiable('x',ini=[0,10*cos(ang_ini*pi/180)])
Differentiable('y',ini=[1,10*sin(ang_ini*pi/180)])
Differentiable('z',ini=[0,0])

Differentiable('qA',ini=[0,0])
Differentiable('qB',ini=[0,0])
Differentiable('qC',ini=[ang_ini*pi/180,0])


Constant(.05,'mC')
Constant(9.81,'g')
Constant(6e-3,'I_11')
Constant(1.292,'rho')
Constant(.1,'Sw')
Constant(.025,'Se')
Constant(.35,'l')
Constant(-.03,'lw')
Constant(.04,'le')
Constant(3*pi/180,'qE')

ini = system.get_ini()

Frame('N')
Frame('A')
Frame('B')
Frame('C')
Frame('E')

system.set_newtonian(N)
A.rotate_fixed_axis_directed(N,[1,0,0],qA)
B.rotate_fixed_axis_directed(A,[0,1,0],qB)
C.rotate_fixed_axis_directed(B,[0,0,1],qC)
E.rotate_fixed_axis_directed(C,[0,0,1],-qE)

pCcm=x*N.x+y*N.y+z*N.z
#pCcm=x*N.x+y*N.y
pCcp=pCcm-lw*C.x

pC1 = pCcm
pC2 = pCcm-l*C.x
pE = pC2-le*E.x
#wNC = N.getw_(C)

vcm = pCcm.time_derivative()

vcp=pCcp.time_derivative()
vcp2 = vcp.dot(vcp)

ve=pE.time_derivative()
ve2 = ve.dot(ve)

IC = Dyadic.build(C,I_11,I_11,I_11)

Body('BodyC',C,pCcm,mC,IC)


vcx = vcp.dot(C.x)
vcy = vcp.dot(-C.y)
angle_of_attack_C = sympy.atan2(vcy,vcx)


vex = ve.dot(E.x)
vey = ve.dot(-E.y)
angle_of_attack_E = sympy.atan2(vey,vex)

#cl = 2*sin(angle_of_attack)*cos(angle_of_attack)
#cd = 2*sin(angle_of_attack)**2

#fl = .5*rho*vcp2*cl*A
#fd = .5*rho*vcp2*cd*A

f_aero_C = rho*vcp2*sympy.sin(angle_of_attack_C)*Sw*C.y
f_aero_E = rho*ve2*sympy.sin(angle_of_attack_E)*Sw*E.y

system.addforcegravity(-g*N.y)
system.addforce(f_aero_C,vcp)
system.addforce(f_aero_E,ve)

points = [pC1,pC2]

#ang = [wNC.dot(C.x),wNC.dot(C.y),wNC.dot(C.z)]

f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma)
states=pynamics.integration.integrate_odeint(func1,ini,t, args=({'constants':system.constant_values},))

#output = Output(ang,system)
#output.calc(states)
#output.plot_time()

po = PointsOutput(points,system)
y=po.calc(states)
#po.plot_time()
#y = y.reshape((-1,2,2))
plt.figure()
for item in y:
    plt.plot(*(item.T),lw=2,marker='o')
#
po.animate(fps = 30, movie_name='body_in_space.mp4',lw=2,marker='o')
