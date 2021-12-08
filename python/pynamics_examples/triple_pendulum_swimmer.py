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
from pynamics.output_points_3d import PointsOutput3D
from pynamics.particle import Particle
from pynamics.constraint import AccelerationConstraint,KinematicConstraint
import pynamics.integration
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
import sympy
sympy.init_printing(pretty_print=False)
import math

system = System()
pynamics.set_system(__name__,system)

constrain_base=False
pp = 30
small = 1e-10

lA = Constant(.1,'lA',system)
lB = Constant(.1,'lB',system)
lC = Constant(.1,'lC',system)

mA = Constant(.1,'mA',system)
mB = Constant(.1,'mB',system)
mC = Constant(.1,'mC',system)

Ixx_A = Constant(.1,'Ixx_A',system)
Iyy_A = Constant(.1,'Iyy_A',system)
Izz_A = Constant(.1,'Izz_A',system)
Ixx_B = Constant(.1,'Ixx_B',system)
Iyy_B = Constant(.1,'Iyy_B',system)
Izz_B = Constant(.1,'Izz_B',system)
Ixx_C = Constant(.1,'Ixx_C',system)
Iyy_C = Constant(.1,'Iyy_C',system)
Izz_C = Constant(.1,'Izz_C',system)

torque = Constant(1e1,'torque',system)
freq = Constant(3e0,'freq',system)

k = Constant(1e2,'k',system)
k2 = Constant(1e0,'k2',system)
k3 = Constant(5e0,'k3',system)
kb = Constant(1e3,'kb',system)
Area = Constant(1,'Area',system)
rho = Constant(1000,'rho')
lS = Constant(.04,'lS',system)
mS = Constant(.2,'mS',system)
r2 = Constant(0,'r2',system)
Ixx_S = Constant(.1,'Ixx_S',system)
Iyy_S = Constant(.1,'Iyy_S',system)
Izz_S = Constant(.1,'Izz_S',system)
paddle_preload = Constant(pp*pi/180,'preload3',system)


x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)
z,z_d,z_dd = Differentiable('z',system)

x2,x2_d,x2_dd = Differentiable('x2',system)
y2,y2_d,y2_dd = Differentiable('y2',system)
z2,z2_d,z2_dd = Differentiable('z2',system)

x3,x3_d,x3_dd = Differentiable('x3',system)
y3,y3_d,y3_dd = Differentiable('y3',system)
z3,z3_d,z3_dd = Differentiable('z3',system)

qA1,qA1_d,qA1_dd = Differentiable('qA1')
qA2,qA2_d,qA2_dd = Differentiable('qA2')
qA3,qA3_d,qA3_dd = Differentiable('qA3')
qB1,qB1_d,qB1_dd = Differentiable('qB1')
qB2,qB2_d,qB2_dd = Differentiable('qB2')
qB3,qB3_d,qB3_dd = Differentiable('qB3')
qC1,qC1_d,qC1_dd = Differentiable('qC1')
qC2,qC2_d,qC2_dd = Differentiable('qC2')
qC3,qC3_d,qC3_dd = Differentiable('qC3')

qS,qS_d,qS_dd = Differentiable('qS')

wAx,wAx_d = Differentiable('wAx',ii = 1)
wAy,wAy_d = Differentiable('wAy',ii = 1)
wAz,wAz_d = Differentiable('wAz',ii = 1)

wBx,wBx_d = Differentiable('wBx',ii = 1)
wBy,wBy_d = Differentiable('wBy',ii = 1)
wBz,wBz_d = Differentiable('wBz',ii = 1)

wCx,wCx_d = Differentiable('wCx',ii = 1)
wCy,wCy_d = Differentiable('wCy',ii = 1)
wCz,wCz_d = Differentiable('wCz',ii = 1)


initialvalues = {}
initialvalues[qA1]=0*pi/180
initialvalues[qA2]=0*pi/180
initialvalues[qA3]=0*pi/180

initialvalues[qB1]=0*pi/180
initialvalues[qB2]=0*pi/180
initialvalues[qB3]=0*pi/180

initialvalues[qC1]=0*pi/180
initialvalues[qC2]=0*pi/180
initialvalues[qC3]=0*pi/180

if not constrain_base:
    initialvalues[x]=0
    initialvalues[x_d]=0
    initialvalues[y]=0
    initialvalues[y_d]=0
    initialvalues[z]=0
    initialvalues[z_d]=0

initialvalues[qS]=0*pi/180
initialvalues[qS_d]=small

initialvalues[wAx]=small
initialvalues[wAy]=small
initialvalues[wAz]=small

initialvalues[wBz]=small

initialvalues[wCz]=small

# initialvalues[qA1_d]=small
# initialvalues[qA2_d]=small
# initialvalues[qA3_d]=small

# initialvalues[qB_d]=small

# initialvalues[qC_d]=small

# initialvalues[wBx]=small
# initialvalues[wBy]=small

# initialvalues[wCx]=small
# initialvalues[wCy]=small

N = Frame('N',system)
A1 = Frame('A1',system)
A2 = Frame('A2',system)
A3 = Frame('A3',system)
B1 = Frame('B1',system)
B2 = Frame('B2',system)
B3 = Frame('B3',system)
C1 = Frame('C1',system)
C2 = Frame('C2',system)
C3 = Frame('C3',system)

S = Frame('S',system)

system.set_newtonian(N)


A1.rotate_fixed_axis(N,[1,0,0],qA1,system)
A2.rotate_fixed_axis(A1,[0,1,0],qA2,system)
A3.rotate_fixed_axis(A2,[0,0,1],qA3,system)
B1.rotate_fixed_axis(N,[1,0,0],qB1,system)
B2.rotate_fixed_axis(B1,[0,1,0],qB2,system)
B3.rotate_fixed_axis(B2,[0,0,1],qB3,system)
C1.rotate_fixed_axis(N,[1,0,0],qC1,system)
C2.rotate_fixed_axis(C1,[0,1,0],qC2,system)
C3.rotate_fixed_axis(C2,[0,0,1],qC3,system)
S.rotate_fixed_axis(A3,[0,0,1],qS,system)

wA1 = N.get_w_to(A3)
wA2 = wAx*A3.x + wAy*A3.y + wAz*A3.z
N.set_w(A3,wA2)

wB1 = N.get_w_to(B3)
wB2 = wBx*B3.x + wBy*B3.y + wBz*B3.z
N.set_w(B3,wB2)

wC1 = N.get_w_to(C3)
wC2 = wCx*C3.x + wCy*C3.y + wCz*C3.z
N.set_w(C3,wC2)

### Vectors
# Define the vectors that describe the kinematics of a series of connected lengths

pAcm=x*A3.x+y*A3.y+z*A3.z
pBase = pAcm-lA/2*A3.x
pAB=pAcm+lA/2*A3.x

pBcm=x2*B3.x+y2*B3.y+z2*B3.z
pBA = pBcm - lB/2*B3.x
pBC = pBcm + lB/2*B3.x

pCcm=x3*C3.x+y3*C3.y+z3*C3.z
pCB = pCcm - lC/2*C3.x
pCtip=pCcm+lC/2*C3.x

pScm = pBase - lS*S.x

# vAcm=pAcm.time_derivative()
# vCcm=pCcm.time_derivative()

va=pAcm.time_derivative()
f_aero_Ax = rho * va.length()*(va.dot(A3.x))*Area/10*A3.x
system.addforce(-f_aero_Ax,va)

f_aero_A = rho * va.length()*(va.dot(A3.y))*Area*A3.y
system.addforce(-f_aero_A,va)

vb=pBcm.time_derivative()
f_aero_B = rho * vb.length()*(vb.dot(B3.y))*Area*B3.y
system.addforce(-f_aero_B,vb)

vctip=pCtip.time_derivative()
f_aero_C = rho * vctip.length()*(vctip.dot(C3.y))*Area*C3.y
system.addforce(-f_aero_C,vctip)

# ## Calculating Velocity
# 
# The angular velocity between frames, and the time derivatives of vectors are extremely useful in calculating the equations of motion and for determining many of the forces that need to be applied to your system (damping, drag, etc).  Thus, it is useful, once kinematics have been defined, to take or find the derivatives of some of those vectors for calculating  linear or angular velocity vectors
# 
# ### Angular Velocity

wA3B3 = A3.get_w_to(B3)
wB3C3 = B3.get_w_to(C3)
wA3S = A3.get_w_to(S)

# ### Define Inertias and Bodies
# The next several lines compute the inertia dyadics of each body and define a rigid body on each frame.  In the case of frame C, we represent the mass as a particle located at point pCcm.  

IA = Dyadic.build(A3,Ixx_A,Iyy_A,Izz_A)
IB = Dyadic.build(B3,Ixx_B,Iyy_B,Izz_B)
IC = Dyadic.build(C3,Ixx_C,Iyy_C,Izz_C)
IS = Dyadic.build(S,Ixx_S,Iyy_S,Izz_S)

BodyA = Body('BodyA',A3,pAcm,mA,IA,system)
BodyB = Body('BodyB',B3,pBcm,mB,IB,system)
BodyC = Body('BodyC',C3,pCcm,mC,IC,system)
BodyS = Body('BodyS',S,pScm,mS,IS,system)

# ## Forces and Torques
# Forces and torques are added to the system with the generic ```addforce``` method.  The first parameter supplied is a vector describing the force applied at a point or the torque applied along a given rotational axis.  The second parameter is the  vector describing the linear speed (for an applied force) or the angular velocity(for an applied torque)

system.addforce(torque*sympy.sin(freq*2*sympy.pi*system.t)*A3.z,wA3S)


# ### Spring Forces
# 
# Spring forces are a special case because the energy stored in springs is conservative and should be considered when calculating the system's potential energy.  To do this, use the ```add_spring_force``` command.  In this method, the first value is the linear spring constant.  The second value is the "stretch" vector, indicating the amount of deflection from the neutral point of the spring.  The final parameter is, as above, the linear or angluar velocity vector (depending on whether your spring is a linear or torsional spring)
# 
# In this case, the torques applied to each joint are dependent upon whether qA, qB, and qC are absolute or relative rotations, as defined above.

# system.add_spring_force1(k,(qA-preload1)*N.z,wNA) 
qAB = -sympy.atan2(A3.x.dot(B3.y),A3.x.dot(B3.x))
system.add_spring_force1(k,(qAB)*A3.z,wA3B3)

qBC = -sympy.atan2(B3.x.dot(C3.y),B3.x.dot(C3.x))
system.add_spring_force1(k,(qBC)*B3.z,wB3C3)
system.add_spring_force1(k2,(qS)*S.z,wA3S)

# ### Gravity
# Again, like springs, the force of gravity is conservative and should be applied to all bodies.  To globally apply the force of gravity to all particles and bodies, you can use the special ```addforcegravity``` method, by supplying the acceleration due to gravity as a vector.  This will get applied to all bodies defined in your system.

#system.addforcegravity(-g*N.y)


# ## Constraints
# Constraints may be defined that prevent the motion of certain elements.  Try uncommenting the commented out line to see what happens.

eq = []
eq.append(A3.z-B3.z)
eq.append(pAB-pBA)
eq.append(B3.z-C3.z)
eq.append(pBC-pCB)
if constrain_base:
    eq.append(pBase-0*N.x)
    eq.append(A3.z-N.z)
eq_d = []
eq_d.extend([item.time_derivative() for item in eq])
eq_d.append(wA1-wA2)
eq_d.append(wB1-wB2)
eq_d.append(wC1-wC2)


eq_dd = [item.time_derivative() for item in eq_d]
eq_dd_scalar = []
eq_dd_scalar.append(eq_dd[0].dot(N.x))
eq_dd_scalar.append(eq_dd[0].dot(N.y))
eq_dd_scalar.append(eq_dd[1].dot(N.x))
eq_dd_scalar.append(eq_dd[1].dot(N.y))
eq_dd_scalar.append(eq_dd[1].dot(N.z))
eq_dd_scalar.append(eq_dd[2].dot(N.x))
eq_dd_scalar.append(eq_dd[2].dot(N.y))
eq_dd_scalar.append(eq_dd[3].dot(N.x))
eq_dd_scalar.append(eq_dd[3].dot(N.y))
eq_dd_scalar.append(eq_dd[3].dot(N.z))
ii=4
if constrain_base:
    eq_dd_scalar.append(eq_dd[4].dot(N.x))
    eq_dd_scalar.append(eq_dd[4].dot(N.y))
    eq_dd_scalar.append(eq_dd[4].dot(N.z))
    eq_dd_scalar.append(eq_dd[5].dot(N.x))
    eq_dd_scalar.append(eq_dd[5].dot(N.y))
    ii=6
eq_dd_scalar.append(eq_dd[ii+0].dot(A2.x))
eq_dd_scalar.append(eq_dd[ii+0].dot(A2.y))
eq_dd_scalar.append(eq_dd[ii+0].dot(A2.z))
eq_dd_scalar.append(eq_dd[ii+1].dot(B2.x))
eq_dd_scalar.append(eq_dd[ii+1].dot(B2.y))
eq_dd_scalar.append(eq_dd[ii+1].dot(B2.z))
eq_dd_scalar.append(eq_dd[ii+2].dot(C2.x))
eq_dd_scalar.append(eq_dd[ii+2].dot(C2.y))
eq_dd_scalar.append(eq_dd[ii+2].dot(C2.z))
    
system.add_constraint(AccelerationConstraint(eq_dd_scalar))



eq_d_scalar = []
eq_d_scalar.append(eq_d[0].dot(N.x))
eq_d_scalar.append(eq_d[0].dot(N.y))
eq_d_scalar.append(eq_d[1].dot(N.x))
eq_d_scalar.append(eq_d[1].dot(N.y))
eq_d_scalar.append(eq_d[1].dot(N.z))
eq_d_scalar.append(eq_d[2].dot(N.x))
eq_d_scalar.append(eq_d[2].dot(N.y))
eq_d_scalar.append(eq_d[3].dot(N.x))
eq_d_scalar.append(eq_d[3].dot(N.y))
eq_d_scalar.append(eq_d[3].dot(N.z))
ii=4
if constrain_base:
    eq_d_scalar.append(eq_d[4].dot(N.x))
    eq_d_scalar.append(eq_d[4].dot(N.y))
    eq_d_scalar.append(eq_d[4].dot(N.z))
    eq_d_scalar.append(eq_d[5].dot(N.x))
    eq_d_scalar.append(eq_d[5].dot(N.y))
    ii=6
eq_d_scalar.append(eq_d[ii+0].dot(A2.x))
eq_d_scalar.append(eq_d[ii+0].dot(A2.y))
eq_d_scalar.append(eq_d[ii+0].dot(A2.z))
eq_d_scalar.append(eq_d[ii+1].dot(B2.x))
eq_d_scalar.append(eq_d[ii+1].dot(B2.y))
eq_d_scalar.append(eq_d[ii+1].dot(B2.z))
eq_d_scalar.append(eq_d[ii+2].dot(C2.x))
eq_d_scalar.append(eq_d[ii+2].dot(C2.y))
eq_d_scalar.append(eq_d[ii+2].dot(C2.z))
eq_d_scalar.append(eq[0].dot(N.x))
eq_d_scalar.append(eq[0].dot(N.y))
eq_d_scalar.append(eq[1].dot(N.x))
eq_d_scalar.append(eq[1].dot(N.y))
eq_d_scalar.append(eq[1].dot(N.z))
eq_d_scalar.append(eq[2].dot(N.x))
eq_d_scalar.append(eq[2].dot(N.y))
eq_d_scalar.append(eq[3].dot(N.x))
eq_d_scalar.append(eq[3].dot(N.y))
eq_d_scalar.append(eq[3].dot(N.z))

kinematic_constraint = KinematicConstraint(eq_d_scalar)
if constrain_base:
    variables = [qA1_d,qA2_d,qA3_d,qB1_d,qB2_d,qB3_d,qC1_d,qC2_d,qC3_d,wBx,wBy,wCx,wCy,x2_d,y2_d,z2_d,x2,y2,z2,x3_d,y3_d,z3_d,x3,y3,z3,x,y,z,x_d,y_d,z_d]
else:
    variables = [qA1_d,qA2_d,qA3_d,qB1_d,qB2_d,qB3_d,qC1_d,qC2_d,qC3_d,wBx,wBy,wCx,wCy,x2_d,y2_d,z2_d,x2,y2,z2,x3_d,y3_d,z3_d,x3,y3,z3]
result = kinematic_constraint.solve_numeric(variables,[1]*len(variables),system.constant_values,initialvalues)
initialvalues.update(result)

f,ma = system.getdynamics()

# # ## Solve for Acceleration
# # 
# # The next line of code solves the system of equations F=ma plus any constraint equations that have been added above.  It returns one or two variables.  func1 is the function that computes the velocity and acceleration given a certain state, and lambda1(optional) supplies the function that computes the constraint forces as a function of the resulting states
# # 
# # There are a few ways of solveing for a.  The below function inverts the mass matrix numerically every time step.  This can be slower because the matrix solution has to be solved for, but is sometimes more tractable than solving the highly nonlinear symbolic expressions that can be generated from the previous step.  The other options would be to use ```state_space_pre_invert```, which pre-inverts the equations symbolically before generating a numerical function, or ```state_space_post_invert2```, which adds Baumgarte's method for intermittent constraints.

static_constants = [rho,lA,lB,lC,lS,mA,mB,mC,mS,r2,Ixx_A,Iyy_A,Izz_A,Ixx_B,Iyy_B,Izz_B,Ixx_A,Iyy_B,Izz_B,Ixx_C,Iyy_C,Izz_C,Ixx_S,Iyy_S,Izz_S]
static_constants = dict([(key,system.constant_values[key]) for key in static_constants])
func1= system.state_space_post_invert(f,ma,return_lambda = False, constants = static_constants)


# %%
statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

# %%
# # ## Integrate
# # 
# # The next line of code integrates the function calculated

tol = 1e-7
tinitial = 0
tfinal = 3
fps = 30
tstep = 1/fps
t = numpy.r_[tinitial:tfinal:tstep]
constants = system.constant_values.copy()
constants[torque]=-1e1
constants[k]=1e2
constants[k2]=2e1
constants[k3]=1e0
constants[kb]=5e0
constants[freq]=3e0
states=pynamics.integration.integrate_odeint(func1,ini,t,rtol=tol,atol=tol, args=({'constants':constants},))

points = [pScm,pBase,pAcm,pAB,pBcm,pBC,pCcm,pCtip]
points_output = PointsOutput3D(points,system)
points_output.calc(states,t)
points_output.plot_time()
#points_output.animate(fps = fps,movie_name = 'triple_pendulum_swimmer.mp4',lw=2,marker='o',color=(1,0,0,1),linestyle='-',azim = -90,elev=90)


