# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
sympy.init_printing(pretty_print=False)

import pynamics
pynamics.integrator = 0
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
from pynamics.constraint import AccelerationConstraint
import pynamics.integration

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(.1,'lA',system)
mA = Constant(.1,'mA',system)

# g = Constant(9.81,'g',system)
freq = Constant(.1,'freq',system)
torque = Constant(10,'torque',system)
Area = Constant(.1,'Area',system)
# b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)
rho = Constant(1000,'rho')

Ixx_motor = Constant(.1,'Ixx_motor')
Iyy_motor = Constant(.1,'Iyy_motor')
Izz_motor = Constant(1,'Izz_motor')

Ixx_plate = Constant(.1,'Ixx_plate')
Iyy_plate = Constant(.1,'Iyy_plate')
Izz_plate = Constant(1,'Izz_plate')

qA,qA_d,qA_dd = Differentiable('qA',system)

initialvalues = {}
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)

pNA=0*N.x
pAcm=pNA+lA/2*A.x
pAtip=pNA+lA*A.x
vAcm=pAcm.time_derivative(N,system)

wNA = N.get_w_to(A)

IA_motor = Dyadic.build(A,Ixx_motor,Iyy_motor,Izz_motor)
IA_plate = Dyadic.build(A,Ixx_plate,Iyy_plate,Izz_plate)
BodyMotor = Body('BodyMotor',A,pNA,mA,IA_motor)
BodyPlate = Body('BodyPlate',A,pAcm,mA,IA_plate)

f_aero_C2 = rho * vAcm.length()*(vAcm.dot(A.y))*Area*A.y
system.addforce(-f_aero_C2,vAcm)
system.add_spring_force1(k,qA*N.z,wNA)

tin = torque*sympy.sin(2*sympy.pi*freq*system.t)
system.addforce(tin*N.z,wNA)


f,ma = system.getdynamics()


changing_constants = [freq]
static_constants = system.constant_values.copy()
for key in changing_constants:
    del static_constants[key]
    
func = system.state_space_post_invert(f,ma,constants=static_constants)

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

points = [pNA,pAcm,pAtip]
points_output = PointsOutput(points,system)
out1 = Output([tin,qA])

my_constants={}
freq_sweep = numpy.r_[-1.5:1:20j]
freq_sweep = 1*10**freq_sweep
amps = []

for ff in freq_sweep:
    tol = 1e-4

    tinitial = 0
    tfinal = 10/ff
    tstep = 1/30
    t = numpy.r_[tinitial:tfinal:tstep]

    my_constants[freq] = ff
    states=pynamics.integration.integrate(func,ini,t,rtol=tol,atol=tol, args=({'constants':my_constants},))

    # points_output.calc(states,t)
    # points_output.animate(fps = 1/tstep,lw=2,movie_name = 'pendulum_in_water.mp4',)
    # points_output.plot_time()
    
    
    out1.calc(states,t)
    # plt.figure()
    # plt.plot(out1.y[:,0],out1.y[:,1])
    amp = out1.y[:,1].max() - out1.y[:,1].min()
    amps.append(amp)

plt.figure()
plt.loglog(freq_sweep,amps)
plt.xlabel('freq')
plt.ylabel('max amplitude')
plt.title('freq vs amplitude')