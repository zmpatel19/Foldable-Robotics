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
from pynamics.constraint import KinematicConstraint

import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(1,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(1,'lC',system)
lD = Constant(1,'lD',system)

m = Constant(1,'m',system)
# mB = Constant(1,'mB',system)
# mC = Constant(1,'mC',system)

# g = Constant(9.81,'g',system)
# b = Constant(1e0,'b',system)
# k = Constant(1e1,'k',system)

tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

# preload1 = Constant(0*pi/180,'preload1',system)
# preload2 = Constant(0*pi/180,'preload2',system)
# preload3 = Constant(0*pi/180,'preload3',system)

# Ixx_A = Constant(1,'Ixx_A',system)
# Iyy_A = Constant(1,'Iyy_A',system)
# Izz_A = Constant(1,'Izz_A',system)
# Ixx_B = Constant(1,'Ixx_B',system)
# Iyy_B = Constant(1,'Iyy_B',system)
# Izz_B = Constant(1,'Izz_B',system)
# Ixx_C = Constant(1,'Ixx_C',system)
# Iyy_C = Constant(1,'Iyy_C',system)
# Izz_C = Constant(1,'Izz_C',system)

qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

initialvalues = {}
initialvalues[qA]=30*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=30*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=150*pi/180
initialvalues[qC_d]=0*pi/180
initialvalues[qD]=-30*pi/180
initialvalues[qD_d]=0*pi/180

statevariables = system.get_state_variables()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(C,[0,0,1],qD,system)

pNA = 0*N.x
pAB = pNA + lA*A.x
pBD = pAB + lB*B.x
pCD = pNA + lC*C.x
pDB = pCD + lD*D.x

points = [pBD,pAB,pNA,pCD,pDB]

statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]


eq = []
eq.append(pBD-pDB)
eq_d = [item.time_derivative() for item in eq]

eq_scalar = []
eq_scalar.append(eq[0].dot(N.x))
eq_scalar.append(eq[0].dot(N.y))

c=KinematicConstraint(eq_scalar)
variables = [qB,qD]
constant_states = list(set(system.get_q(0))-set(variables))

constants = system.constant_values.copy()
for key in constant_states:
    constants[key] = initialvalues[key] 
guess = [initialvalues[item] for item in variables]
result = c.solve_numeric(variables,guess,constants)


ini = []
for item in system.get_state_variables():
    if item in variables:
        ini.append(result[item])
    else:
        ini.append(initialvalues[item])
        
points = PointsOutput(points, constant_values=system.constant_values)
points.calc(numpy.array([ini0,ini]),[0,1])
points.plot_time()


# eq_d = sympy.Matrix(eq_d)
# qi = sympy.Matrix([qA_d])
# qd = sympy.Matrix([qB_d,qC_d])

# C = eq_d.jacobian(qi)
# D = eq_d.jacobian(qd)

# J = -D.inv()*C
# J.simplify()

# qd2 = J*qi

# subs = dict([(ii,jj) for ii,jj in zip(qd,qd2)])


pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pNA+lC/2*C.x
pDcm=pCD+lD/2*D.x

# # wNA = N.get_w_to(A)
# # wAB = A.get_w_to(B)
# # wBC = B.get_w_to(C)
wND = N.get_w_to(D)

# # IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
# # IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
# # IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

# # BodyA = Body('BodyA',A,pAcm,mA,IA,system)
# # BodyB = Body('BodyB',B,pBcm,mB,IB,system)
# # #BodyC = Body('BodyC',C,pCcm,mC,IC,system)
BodyA = Particle(pAcm,m,'ParticleA',system)
BodyB = Particle(pBcm,m,'ParticleB',system)
BodyC = Particle(pCcm,m,'ParticleC',system)
BodyD = Particle(pDcm,m,'ParticleD',system)

# # system.addforce(-b*wNA,wNA)
# # system.addforce(-b*wAB,wAB)
# # system.addforce(-b*wBC,wBC)

# # system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
# # system.add_spring_force1(k,(qB-preload2)*A.z,wAB)
# # system.add_spring_force1(k,(qC-preload3)*B.z,wBC)

# # system.addforcegravity(-g*N.y)

# # vCtip = pCtip.time_derivative(N,system)




# # f,ma = system.getdynamics()
# # dyn = sympy.Matrix(f)-sympy.Matrix(ma)
# # eq_dd = sympy.Matrix(eq_dd)


# # A_new = A+(B*D.inv()*C)
# # func1 = system.state_space_post_invert(f,ma,eq_dd)
# # states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14, args=({'constants':system.constant_values},))

# # KE = system.get_KE()
# # PE = system.getPEGravity(pNA) - system.getPESprings()

# # points = [pNA,pAB,pBC,pCtip]
# # #points = [item for item2 in points for item in [item2.dot(system.newtonian.x),item2.dot(system.newtonian.y)]]
# # points_output = PointsOutput(points,system)
# # y = points_output.calc(states,t)
# # #y.resize(y.shape[0],int(y.shape[1]/2),2)

# # plt.figure()
# # plt.plot(t,states[:,:3])

# # plt.figure()
# # plt.plot(*(y[::int(len(y)/20)].T))
# # plt.axis('equal')

# # energy_output = Output([KE-PE],system)
# # energy_output.calc(states,t)

# # plt.figure()
# # plt.plot(energy_output.y)

# # points_output.animate(fps = 30,movie_name = 'render.mp4',lw=2)
# # #a()
