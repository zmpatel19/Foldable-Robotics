# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
confirmed against:https://www.ee.usyd.edu.au/tutorials_online/matlab/examples/motor/motor.html
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
import pynamics.motor
#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi




system = System()

L = Constant(name='L',system=system)
V = Constant(name='V',system=system)
R = Constant(name='R',system=system)
Im = Constant(name='Im',system=system)
Il = Constant(name='Il',system=system)
G = Constant(name='G',system=system)
b = Constant(name='b',system=system)
kv = Constant(name='kv',system=system)
kt = Constant(name='kt',system=system)
Tl = Constant(name='Tl',system=system)
m = Constant(name='m',system=system)
g = Constant(name='g',system=system)

tinitial = 0
tfinal = 3
tstep = .01
t = numpy.r_[tinitial:tfinal:tstep]

qB,qB_d,qB_dd = Differentiable('qB',system)
i,i_d= Differentiable('i',ii=1,system=system)


constants = {}
constants[L] = .5
constants[V] = 1
constants[R] = 1
constants[G] = 10
constants[Im] = .01
constants[Il] = .1
constants[b] = .1
constants[kv] = .01
constants[kt] = .01
constants[Tl] = 0
constants[m] = 1
constants[g] = 9.81

initialvalues = {}
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[i]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
B = Frame('B')

system.set_newtonian(N)
B.rotate_fixed_axis_directed(N,[0,0,1],qB,system)

pO = 0*N.x
wNB = N.getw_(B)

I_load = Dyadic.build(B,Il,Il,Il)
Load = Body('Load',B,pO,m,I_load,system)

axis = N.z
#T = kt*(V/R)
#T = kt*(V/R)-kv*G*qB_d
#system.addforce(T*axis,G*qB_d*axis)
system.addforce(-Tl*B.z,wNB)
eq_d = []
eq_dd= [system.derivative(item) for item in eq_d]

pynamics.motor.add_motor_dynamics(B,Im,G,L,i,i_d,kt,kv,b,V,R,qB_d,axis,system)
f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,eq_dd)
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=1e-10,atol=1e-10,args=({'constants':constants,'alpha':1e2,'beta':1e1},))

# =============================================================================
KE = system.get_KE()
PE = system.getPEGravity(0*N.x) - system.getPESprings()
energy = Output([KE-PE], constant_values = constants)
energy.calc(states)
energy.plot_time()
# =============================================================================

positions = Output(system.get_q(0), constant_values = constants)
positions.calc(states)
positions.plot_time()

speeds = Output(system.get_q(1), constant_values = constants)
speeds.calc(states)
speeds.plot_time()

y= Output([G*qB_d], constant_values=constants)
y.calc(states)
y.plot_time()

