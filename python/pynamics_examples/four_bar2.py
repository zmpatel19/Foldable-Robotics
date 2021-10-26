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
from pynamics.constraint import AccelerationConstraint,KinematicConstraint

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

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)
tol = 1e-5
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

# T1 = Constant(1,'T1',system)
# T2 = Constant(1,'T2',system)

T1 = sympy.Symbol('T1')
T2 = sympy.Symbol('T2')
T3 = sympy.Symbol('T3')
T4 = sympy.Symbol('T4')

# Fconst = sympy.Symbol('Fconst')

Fx_tip = sympy.Symbol('Fx_tip')
Fy_tip= sympy.Symbol('Fy_tip')
T_tip= sympy.Symbol('T_tip')


qA,qA_d,qA_dd = Differentiable('qA',system)
qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

initialvalues = {}
initialvalues[qA]=60*pi/180
initialvalues[qA_d]=0*pi/180
initialvalues[qB]=30*pi/180
initialvalues[qB_d]=0*pi/180
initialvalues[qC]=120*pi/180
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
D.rotate_fixed_axis(B,[0,0,1],qD,system)

pNA=0*N.x
pAB=pNA+lA*A.x
pBD = pAB + lB*B.x

pNC=pNA
pCD = pNC+lC*C.x
pDB = pCD + lD*D.x

vCD_AB = pAB-pCD
uCD_AB = 1/(vCD_AB.length()) * vCD_AB

vCD=pCD.time_derivative()
vAB=pAB.time_derivative()

points = [pDB,pCD,pNC,pNA,pAB,pBD]

statevariables = system.get_state_variables()
ini0 = [initialvalues[item] for item in statevariables]


eq = []
eq.append((pBD-pDB).dot(N.x))
eq.append((pBD-pDB).dot(N.y))
eq_d=[(system.derivative(item)) for item in eq]
eq_dd=[(system.derivative(item)) for item in eq_d]

c=KinematicConstraint(eq)

variables = [qA,qB,qC,qD]
guess = [initialvalues[item] for item in variables]
result = c.solve_numeric(variables,guess,system.constant_values)

ini = []
for item in system.get_state_variables():
    if item in variables:
        ini.append(result[item])
    else:
        ini.append(initialvalues[item])
        
po1 = PointsOutput(points, constant_values=system.constant_values)
po1.calc(numpy.array([ini0,ini]),[0,1])
po1.plot_time()




pAcm=pNA+lA/2*A.x
pBcm=pAB+lB/2*B.x
pCcm=pNC+lC/2*C.x
pDcm=pCD+lD/2*D.x

vBD= pBD.time_derivative()

wNA = N.get_w_to(A)
wAB = A.get_w_to(B)
wNC = N.get_w_to(C)
wCD = C.get_w_to(D)

wNB = N.get_w_to(B)

# IA = Dyadic.build(A,Ixx_A,Iyy_A,Izz_A)
# IB = Dyadic.build(B,Ixx_B,Iyy_B,Izz_B)
# IC = Dyadic.build(C,Ixx_C,Iyy_C,Izz_C)

# BodyA = Body('BodyA',A,pAcm,mA,IA,system)
# BodyB = Body('BodyB',B,pBcm,mB,IB,system)
# #BodyC = Body('BodyC',C,pCcm,mC,IC,system)
BodyA = Particle(pAcm,m,'ParticleA',system)
BodyB = Particle(pBcm,m,'ParticleB',system)
BodyC = Particle(pCcm,m,'ParticleC',system)
BodyD = Particle(pDcm,m,'ParticleD',system)

system.addforce(-b*wNA,wNA)
system.addforce(-b*wNC,wNC)

system.addforce(T2*uCD_AB,vCD)
system.addforce(-T2*uCD_AB,vAB)
system.addforce(-T2*N.y,vAB)
system.addforce(-T1*N.y,vAB)

system.addforce(-T3*uCD_AB,vAB)
system.addforce(T3*uCD_AB,vCD)
system.addforce(-T3*N.y,vCD)
system.addforce(-T4*N.y,vCD)

system.addforce(Fx_tip*N.x,vBD)
system.addforce(Fy_tip*N.y,vBD)
system.addforce(T_tip*N.z,wNB)

# system.addforce(-b*wAB,wAB)
# system.addforce(-b*wBC,wBC)

# system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
# system.add_spring_force1(k,(qB-preload2)*A.z,wAB)
# system.add_spring_force1(k,(qC-preload3)*B.z,wBC)

system.addforcegravity(-g*N.y)

# vCtip = pCtip.time_derivative(N,system)


eq = []
eq.append((pBD-pDB))
eq.append((pCD-pAB))
eq.append((pAB))
eq_d = []
eq_d.extend([item.time_derivative() for item in eq])
# eq_d_scalar = []
# eq_d_scalar.append(eq_d[0].dot(N.x))
# eq_d_scalar.append(eq_d[0].dot(N.y))
eq_dd = [item.time_derivative() for item in eq_d]
eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(N.x))
eq_dd_scalar.append(eq_dd[0].dot(N.y))
eq_dd_scalar.append(eq_dd[1].dot(N.x))
eq_dd_scalar.append(eq_dd[2].dot(N.y))
system.add_constraint(AccelerationConstraint(eq_dd_scalar))



f,ma = system.getdynamics()
# dyn = sympy.Matrix(f)-sympy.Matrix(ma)
# eq_dd = sympy.Matrix(eq_dd)

# eq_d_scalar = sympy.Matrix(eq_d_scalar)
# qi = sympy.Matrix([qA_d,qC_d])
# qd = sympy.Matrix([qB_d,qD_d])

# JC = eq_d_scalar.jacobian(qi)
# JD = eq_d_scalar.jacobian(qd)

# J = -JD.inv()*JC
# J.simplify()
# qd_subs = J*qi
# qd2 = J*qi

# subs = dict([(ii,jj) for ii,jj in zip(qd,qd2)])

# # A_new = A+(B*D.inv()*C)
# func1 = system.state_space_post_invert(f,ma)
# states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol, args=({'constants':system.constant_values},))

# KE = system.get_KE()
# PE = system.getPEGravity(pNA) - system.getPESprings()

# # points = [pNA,pAB,pBC,pCtip]
# #points = [item for item2 in points for item in [item2.dot(system.newtonian.x),item2.dot(system.newtonian.y)]]
# points_output = PointsOutput(points,system)
# y = points_output.calc(states,t)
# #y.resize(y.shape[0],int(y.shape[1]/2),2)

# plt.figure()
# plt.plot(t,states[:,:3])

# plt.figure()
# plt.plot(*(y[::int(len(y)/20)].T))
# plt.axis('equal')

# energy_output = Output([KE-PE],system)
# energy_output.calc(states,t)

# plt.figure()
# plt.plot(energy_output.y)

# points_output.animate(fps = 30,movie_name = 'render.mp4',lw=2)
# #a()


fm = sympy.Matrix(f)
fm = fm.subs(initialvalues)
# fm = fm.subs({qA_d:0,qB_d:0,qC_d:0,qD_d:0})
fm = fm.subs(system.constant_values)
# fm = fm.subs(dict(zip(qd,qd_subs)))

result = sympy.solve(fm[:], [T1,T2,T3,T4])

vx = vBD.dot(N.x)
vy = vBD.dot(N.y)
wNB_scalar = wNB.dot(N.z)


q_d = sympy.Matrix([qA_d,qB_d,qC_d,qD_d])
q_ind = sympy.Matrix([qA_d,qC_d])
q_dep = sympy.Matrix([qB_d,qD_d])
v = sympy.Matrix([vx,vy,wNB_scalar])
J = v.jacobian(q_d)
J_ind = v.jacobian(q_ind)
J_dep = v.jacobian(q_dep)

zero  = J_ind*q_ind+J_dep*q_dep - J*q_d

eq = pBD-pDB
eq_d = eq.time_derivative()
eq_d_scalar = []
eq_d_scalar.append(eq_d.dot(N.x))
eq_d_scalar.append(eq_d.dot(N.y))
eq_d_scalar= sympy.Matrix(eq_d_scalar)
J_constraint = eq_d_scalar.jacobian(q_d)
A= eq_d_scalar.jacobian(q_ind)
B= eq_d_scalar.jacobian(q_dep)

C = -B.inv()*A

J_new = (J_ind+J_dep*C)
J_new = J_new.subs(initialvalues)
J_new = J_new.subs(system.constant_values)
# zero2 = J_ind*q_ind+J_dep*C*q_ind - J*sympy.Matrix(q_d)
zero2 = J_new*q_ind - J*q_d

f = sympy.Matrix([Fx_tip,Fy_tip,T_tip])

T = J.T*f