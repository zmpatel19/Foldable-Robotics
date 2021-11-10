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
plt.close('all')
plt.ion()

from math import pi
system = System()
pynamics.set_system(__name__,system)

lA = Constant(1,'lA',system)
# lA = Constant(1,'lA',system)
# lA = Constant(1,'lA',system)
# lA = Constant(1,'lA',system)

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
angle_value = 60
initialvalues[qA]   =(angle_value)*pi/180
initialvalues[qA_d] =0*pi/180
initialvalues[qB]   =pi-2*(angle_value)*pi/180
initialvalues[qB_d] =0*pi/180
initialvalues[qC]   =pi - angle_value*pi/180
initialvalues[qC_d] =0*pi/180
initialvalues[qD]   =2*angle_value*pi/180 -pi
initialvalues[qD_d] =0*pi/180

# initialvalues[qA]   =60*pi/180
# initialvalues[qA_d] =0*pi/180
# initialvalues[qB]   =60*pi/180
# initialvalues[qB_d] =0*pi/180
# initialvalues[qC]   =120*pi/180
# initialvalues[qC_d] =0*pi/180
# initialvalues[qD]   =-60*pi/180
# initialvalues[qD_d] =0*pi/180

statevariables = system.get_state_variables()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
D = Frame('D',system)

# V1 = Frame('V_1',system)
# V2 = Frame('V_2',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],qA,system)
B.rotate_fixed_axis(A,[0,0,1],qB,system)
C.rotate_fixed_axis(N,[0,0,1],qC,system)
D.rotate_fixed_axis(B,[0,0,1],qD,system)


pNA=0*N.x
pAB=pNA+lA*A.x
pBD = pAB + lA*B.x

pNC=pNA
pCD = pNC+lA*C.x
pDB = pCD + lA*D.x

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

# po1.calc(numpy.array([ini,ini]),[0,1])
# po1.plot_time()


pAcm = pNA+lA/2*A.x
pBcm = pAB+lA/2*B.x
pCcm = pNC+lA/2*C.x
pDcm = pCD+lA/2*D.x

vBD = pBD.time_derivative()
vDB = pDB.time_derivative()

wNA = N.get_w_to(A)
wAB = A.get_w_to(B)
wNC = N.get_w_to(C)
wCD = C.get_w_to(D)

wNB = N.get_w_to(B)

uBD = 1/(vBD.length())*vBD
uDB = 1/(vDB.length())*vDB

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

system.addforce(T3*uCD_AB,vCD)
system.addforce(-T3*uCD_AB,vAB)
system.addforce(-T3*N.y,vCD)
system.addforce(-T4*N.y,vCD)

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


fm = sympy.Matrix(f)
fm = fm.subs(initialvalues)
fm = fm.subs({qA_d:0,qB_d:0,qC_d:0,qD_d:0})
fm = fm.subs(system.constant_values)


result = sympy.solve(fm[:], [T1,T2,T3,T4])

vx = vBD.dot(N.x)
vy = vBD.dot(N.y)
wNB_scalar = wNB.dot(N.z)

q_d = sympy.Matrix([qA_d,qB_d,qC_d,qD_d])
q_ind = sympy.Matrix([qA_d,qC_d])
q_dep = sympy.Matrix([qB_d,qD_d])


v = sympy.Matrix([vx,vy])
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
A_m= eq_d_scalar.jacobian(q_ind)
B_m= eq_d_scalar.jacobian(q_dep)

C_m = -B_m.inv()*A_m

J_new = (J_ind+J_dep*C_m)

f = sympy.Matrix([Fx_tip,Fy_tip,T_tip])

l_3 = (pAB-pCD)
l_4 = (pCD-pAB)

l_3_length = (l_3.dot(l_3))**0.5
l_4_length = (l_4.dot(l_4))**0.5

pV3_0 = pAB - 0.5*lA*N.y - l_3_length*N.y
pV4_0 = pCD - 0.5*lA*N.y - l_4_length*N.y

pV5_0 = pAB - l_3_length*N.x
pV6_0 = pCD - l_4_length*N.x

pV1_0 = pAB - lA*N.y
pV2_0 = pCD - lA*N.y

v_l1 = pV1_0.time_derivative().dot(N.y)
v_l2 = pV2_0.time_derivative().dot(N.y)
v_l3 = pV3_0.time_derivative().dot(N.y)
v_l4 = pV4_0.time_derivative().dot(N.y)
# v_l5 = pV5_0.time_derivative().dot(N.x)
# v_l6 = pV6_0.time_derivative().dot(N.x)

# v_t = sympy.Matrix([v_l1,v_l3,v_l2,v_l4,v_l5,v_l6])
v_t = sympy.Matrix([v_l1,v_l2,v_l3,v_l4])

J_t  = v_t.jacobian(q_d)
J_t_ind = v_t.jacobian(q_ind)

J_new_inv = J_new.inv()

f1 = sympy.Matrix([Fx_tip,Fy_tip])

f_t = (J_t_ind* J_new_inv)*f1

cond1 = {}
cond1[lA] = 0.05
cond1[Fx_tip] = 0
cond1[Fy_tip] = 0
# cond1[T_tip] = 10

initialvalues = {}
angle_value = 45
initialvalues[qA]   =(angle_value)*pi/180
initialvalues[qA_d] =0*pi/180
initialvalues[qB]   =pi-2*(angle_value)*pi/180
initialvalues[qB_d] =0*pi/180
initialvalues[qC]   =pi - angle_value*pi/180
initialvalues[qC_d] =0*pi/180
initialvalues[qD]   =2*angle_value*pi/180 -pi
initialvalues[qD_d] =0*pi/180


f_t_num = f_t.subs(initialvalues)
f_t_num1 = f_t_num.subs(cond1)

print(f_t_num1)