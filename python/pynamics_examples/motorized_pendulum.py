# -*- coding: utf-8 -*-
"""
confirmed against:https://www.ee.usyd.edu.au/tutorials_online/matlab/examples/motor/motor.html
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

'''This code uses motor equations using the input shaft of a gear motor as the reference for motor speeds.  this prevents two frames from eneding to be defined.'''

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle, PseudoParticle
import pynamics.integration
from pynamics.constraint import AccelerationConstraint

#import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi

system = System()
pynamics.set_system(__name__,system)

L = Constant(name='L',system=system)
V = Constant(name='V',system=system)
R = Constant(name='R',system=system)
Im = Constant(name='Im',system=system)
Il = Constant(name='Il',system=system)
Ib = Constant(name='Ib',system=system)
G = Constant(name='G',system=system)
b = Constant(name='b',system=system)
kv = Constant(name='kv',system=system)
kt = Constant(name='kt',system=system)
m_pendulum = Constant(name='m_pendulum',system=system)
m_motor = Constant(name='m_motor',system=system)
m_body = Constant(name='m_body',system=system)
g = Constant(name='g',system=system)
l = Constant(name='l',system=system)


x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)
qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qM,qM_d,qM_dd = Differentiable('qM',system)
#wB,wB_d= Differentiable('wB',ii=1,limit=3,system=system)
i,i_d= Differentiable('i',ii=1,system=system)


constants = {}
constants[L] = .541e-3
constants[V] = 12
constants[R] = 2.29
constants[G] = 10
constants[Im] = 52.3*1e-3*(1e-2)**2
constants[Il] = .1
constants[Ib] = .1
constants[b] = 0
constants[kv] = .01
constants[kt] = .01
constants[m_pendulum] = .1
constants[m_motor] = .1
constants[m_body] = .1
constants[g] = 9.81
constants[l] = .01

initialvalues = {}
initialvalues[x]=0*pi/180
initialvalues[x_d]=0*pi/180
initialvalues[y]=0*pi/180
initialvalues[y_d]=0*pi/180
initialvalues[qA]=0*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=0*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qM]=0*pi/180
initialvalues[qM_d]=0*pi/180
#initialvalues[wB]=0*pi/180
#initialvalues[a]=0
initialvalues[i]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
M = Frame('M',system)
Z = Frame('Z',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
M.rotate_fixed_axis(A,[0,0,1],qM,system)

pO = 0*N.x
pAcm = x*N.x+y*N.y
wNA = N.get_w_to(A)
wAB = A.get_w_to(B)
wAM = A.get_w_to(M)
pBcm = pAcm+l*B.x

vAcm = pAcm.time_derivative()
#wNA = G*wNB
#aNA = wNA.time_derivative()
#wNB = wB*B.z
#aNB = wB_d*B.z

I_motor = Dyadic.build(M,Im,Im,Im)
I_body = Dyadic.build(A,Ib,Ib,Ib)
I_load = Dyadic.build(B,Il,Il,Il)

Motor = Body('Motor',M,pAcm,0,I_motor,system)
main_body= Body('main_body',A,pAcm,m_body,I_body,system)
Load = Body('Load',B,pBcm,m_pendulum,I_load,system)

Inductor = PseudoParticle(0*Z.x,L,name='Inductor',vCM = i*Z.x,aCM = i_d*Z.x)

#Load = Body('Load',B,pO,0,I_load,system,wNBody = wNB,alNBody = aNB)

#T = kt*(V/R)-kv*G*qB_d
T = kt*i
system.addforce(T*A.z,wAM)
system.addforce((V-i*R - kv*G*qB_d)*Z.x,i*Z.x)

system.addforce(-b*wAM,wAM)
system.addforcegravity(-g*N.y)

# eq_d = []
# eq = [pAcm]
eq = []
eq_d= [item.time_derivative() for item in eq]
eq_d.append(wAM - G*wAB)
eq_d.append(vAcm)
eq_d.append(wNA)
eq_dd= [item.time_derivative() for item in eq_d]
eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(A.z))
eq_dd_scalar.append(eq_dd[1].dot(N.x))
eq_dd_scalar.append(eq_dd[1].dot(N.y))
eq_dd_scalar.append(eq_dd[2].dot(N.z))
system.add_constraint(AccelerationConstraint(eq_dd_scalar))

#import sympy
#ind = sympy.Matrix([wB])
#dep = sympy.Matrix([qA_d])
#
#EQ = sympy.Matrix(eq_d)
#A = EQ.jacobian(ind)
#B = EQ.jacobian(dep)
#dep2 = sympy.simplify(B.solve(-(A),method = 'LU'))

f,ma = system.getdynamics()
#f,ma = system.getdynamics([qB_d])
#f,ma = system.getdynamics([qA_d,wB])
#f.append(V-i*R - kv*G*qB_d)
#ma.append(L*i_d )
#res = system.solve_f_ma(f,ma,system.get_q(2))
#func1 = system.state_space_pre_invert(f,ma,constants = system.constant_values)
func1 = system.state_space_post_invert(f,ma)

tinitial = 0
tfinal = 3
fps = 30
tstep = 1/fps
t = numpy.r_[tinitial:tfinal:tstep]

tol = 1e-5
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol,args=({'constants':constants},))

# # =============================================================================
# KE = system.get_KE()
# PE = system.getPEGravity(0*N.x) - system.getPESprings()
# energy = Output([KE-PE], constant_values = constants)
# energy.calc(states,t)
# energy.plot_time()
# # =============================================================================

# positions = Output(system.get_q(0), constant_values = constants)
# positions.calc(states,t)
# positions.plot_time()

# speeds = Output(system.get_q(1), constant_values = constants)
# speeds.calc(states,t)
# speeds.plot_time()

# y= Output([G*qB_d], constant_values=constants)
# y.calc(states,t)
# y.plot_time()

points = [pAcm,pBcm]
po = PointsOutput(points,constant_values=constants)
po.calc(states,t)
po.plot_time()
po.animate(fps = fps,movie_name = 'triple_pendulum.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-')
