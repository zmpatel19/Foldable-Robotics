# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.tic()
#pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle
import pynamics.inertia

import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

error = 1e-3
error_tol = 1e-3

alpha = 1e6
beta = 1e5

#preload1 = Constant('preload1',0*pi/180,system)
g = Constant('g',9.81,system)

k_scaling = 3
b_scaling = 2

k_hip = Constant('k_hip',1*10**k_scaling,system)
k_knee = Constant('k_knee',1*10**k_scaling,system)
k_ankle = Constant('k_ankle',1*10**k_scaling,system)
k_foot = Constant('k_foot',1*10**k_scaling,system)

b_hip = Constant('b_hip',1*10**b_scaling,system)
b_knee = Constant('b_knee',1*10**b_scaling,system)
b_ankle = Constant('b_ankle',1*10**b_scaling,system)
b_foot = Constant('b_foot',1*10**b_scaling,system)

l_femur = Constant('l_femur',.3,system)
l_tibia = Constant('l_tibia',.5,system)
l_tmt = Constant('l_tmt',.3,system)
l_foot = Constant('l_foot',.2,system)

l_bcm_hip_x = Constant('l_bcm_hip_x',-.05,system)
l_bcm_hip_y = Constant('l_bcm_hip_y',-.05,system)

l_bcm_tail_x = Constant('l_bcm_tail_x',-.1,system)
l_bcm_tail_y = Constant('l_bcm_tail_y',-.1,system)

q_hip_preload = Constant('q_hip_preload',90*pi/180,system)
q_knee_preload = Constant('q_knee_preload',-90*pi/180,system)
q_ankle_preload = Constant('q_ankle_preload',90*pi/180,system)
q_foot_preload = Constant('q_foot_preload',-45*pi/180,system)

r_body = 1
density = 1000
I_body_principal,m_body,v_body = pynamics.inertia.solid_ellipsoid(r_body,density = density)

I_body_xx = Constant('I_body_xx',I_body_principal[0],system)
I_body_yy = Constant('I_body_yy',I_body_principal[1],system)
I_body_zz = Constant('I_body_zz',I_body_principal[2],system)

r_foot = .01
I_foot_principal,m_foot,v_foot = pynamics.inertia.solid_ellipsoid(r_foot,density = density)

I_foot_xx = Constant('I_foot_xx',I_foot_principal[0],system)
I_foot_yy = Constant('I_foot_yy',I_foot_principal[1],system)
I_foot_zz = Constant('I_foot_zz',I_foot_principal[2],system)

m_body = Constant('m_femur',m_body,system)
m_femur = Constant('m_femur',.1,system)
m_tibia = Constant('m_tibia',.1,system)
m_tmt = Constant('m_tmt',.1,system)
m_foot = Constant('m_foot',m_foot,system)

tinitial = 0
tfinal = 5
tstep = .01
t = numpy.r_[tinitial:tfinal:tstep]

x_body,x_body_d,x_body_dd = Differentiable(system,'x_body')
y_body,y_body_d,y_body_dd = Differentiable(system,'y_body')
q_body,q_body_d,q_body_dd = Differentiable(system,'q_body')

q_hip,q_hip_d,q_hip_dd = Differentiable(system,'q_hip')
q_knee,q_knee_d,q_knee_dd = Differentiable(system,'q_knee')
q_ankle,q_ankle_d,q_ankle_dd = Differentiable(system,'q_ankle')
q_foot,q_foot_d,q_foot_dd = Differentiable(system,'q_foot')
#q_tail,q_tail_d,q_tail_dd = Differentiable(system,'q_tail')

initialvalues = {}

initialvalues[q_body]=-45*pi/180
initialvalues[q_body_d]=0

initialvalues[x_body]=0
initialvalues[x_body_d]=0

initialvalues[y_body]=2
initialvalues[y_body_d]=0

initialvalues[q_hip]=0*pi/180
#initialvalues[q_hip]=45*pi/180
initialvalues[q_hip_d]=0

initialvalues[q_knee]=0*pi/180
#initialvalues[q_knee]=-45*pi/180
initialvalues[q_knee_d]=0

initialvalues[q_ankle]=0*pi/180
#initialvalues[q_ankle]=45*pi/180
initialvalues[q_ankle_d]=0

initialvalues[q_foot]=0
initialvalues[q_foot_d]=0

#initialvalues[q_tail]=0
#initialvalues[q_tail_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
F_body= Frame('F_body')
F_femur= Frame('F_femur')
F_tibia= Frame('F_tibia')
F_tmt= Frame('F_tmt')
F_foot= Frame('F_foot')

system.set_newtonian(N)
F_body.rotate_fixed_axis_directed(N,[0,0,1],q_body,system)
F_femur.rotate_fixed_axis_directed(N,[0,0,1],q_hip,system)
F_tibia.rotate_fixed_axis_directed(N,[0,0,1],q_knee,system)
F_tmt.rotate_fixed_axis_directed(N,[0,0,1],q_ankle,system)
F_foot.rotate_fixed_axis_directed(N,[0,0,1],q_foot,system)

pOrigin = 0*N.x
p_body_cm= x_body*N.x +y_body*N.y
p_hip = p_body_cm + l_bcm_hip_x*F_body.x + l_bcm_hip_y*F_body.y 
p_knee = p_hip - l_femur*F_femur.y
p_ankle = p_knee - l_tibia*F_tibia.y
p_foot = p_ankle - l_tmt*F_tmt.y
p_toes1 = p_foot+l_foot/2*F_foot.x
p_toes2 = p_foot-l_foot/2*F_foot.x

p_femur = p_hip - l_femur/2*F_femur.y
p_tibia = p_knee - l_tibia/2*F_tibia.y
p_tmt = p_ankle - l_tmt/2*F_tmt.y

I_BirdBody = Dyadic.build(F_body,I_body_xx,I_body_yy,I_body_zz)
BirdBody= Body('BirdBody',F_body,p_body_cm,m_body,I_BirdBody,system)

Particle_femur = Particle(system,p_femur,m_femur,'Particle_femur')
Particle_tibia = Particle(system,p_tibia,m_tibia,'Particle_tibia')
Particle_tmt = Particle(system,p_tmt,m_tmt,'Particle_tmt')
#Particle_foot = Particle(system,p_foot,m_foot,'Particle_foot')

I_Foot = Dyadic.build(F_foot,I_foot_xx,I_foot_yy,I_foot_zz)
FootBody= Body('FootBody',F_foot,p_foot,m_foot,I_Foot,system)

#vpmass_body = pmass_body.time_derivative(N,system)
#vpmass_femur = pmass_femur.time_derivative(N,system)

#l_ = pmass_body-pmass_femur
#l = (l_.dot(l_))**.5
#l_d =system.derivative(l)
#stretch = l - l0
#ul_ = l_*(l**-1)
#vl = l_.time_derivative(N,system)

#system.add_spring_force(k,stretch*ul_,vl)
#system.addforce(-k*stretch*ul_,vpmass_body)
#system.addforce(k*stretch*ul_,vpmass_femur)

#system.addforce(-b*l_d*ul_,vpmass_body)
#system.addforce(b*l_d*ul_,vpmass_femur)

w_hip = F_body.getw_(F_femur)
w_knee = F_femur.getw_(F_tibia)
w_ankle = F_tibia.getw_(F_tmt)
w_foot = F_tmt.getw_(F_foot)

system.add_spring_force(k_hip,(q_hip-q_body-q_hip_preload)*F_body.z,w_hip) 
system.add_spring_force(k_knee,(q_knee-q_hip-q_knee_preload)*F_femur.z,w_knee) 
system.add_spring_force(k_ankle,(q_ankle-q_knee-q_ankle_preload)*F_tibia.z,w_ankle) 
system.add_spring_force(k_foot,(q_foot-q_ankle-q_foot_preload)*F_tmt.z,w_foot) 

system.addforce(-b_hip*w_hip,w_hip)
system.addforce(-b_knee*w_knee,w_knee)
system.addforce(-b_ankle*w_ankle,w_ankle)
system.addforce(-b_foot*w_foot,w_foot)

#system.addforce(k*l*ul_,vpmass_femur)
#system.addforce(-b*vl,vl)
#system.addforce(-b*vl,vl)
#system.addforce(-b*vl,vl)



system.addforcegravity(-g*N.y)

#system.addforcegravity(-g*N.y)
#system.addforcegravity(-g*N.y)


eq1 = []
eq1.append(x_body)
eq1.append(y_body)
eq1.append(q_body)
#eq1.append(0-p_toes1.dot(N.y))
#eq1.append(0-p_toes2.dot(N.y))
#eq1.append(0-p_toes1.dot(N.x))
#eq1.append(0-p_toes2.dot(N.x))
eq1_d=[system.derivative(item) for item in eq1]
eq1_dd=[system.derivative(system.derivative(item)) for item in eq1]

#a = []
#a.append(0-p_toes1.dot(N.y))
#a.append(0-p_toes2.dot(N.y))
#a.append(0-p_toes1.dot(N.y))
#a.append(0-p_toes2.dot(N.y))
#b = [(item+abs(item)) for item in a]

#x_body = BodyA.pCM.dot(N.y)
#x2 = Particle2.pCM.dot(N.y)

KE = system.KE
PE = system.getPEGravity(pOrigin) - system.getPESprings()


points = []
points.append(p_body_cm)
points.append(p_hip)
points.append(p_knee)
points.append(p_ankle)
points.append(p_foot)
points.append(p_toes1)
points.append(p_toes2)

outputlist = [item for point in points for item in [point.dot(item2) for item2 in [N.x,N.y]]]

print('solving dynamics...')
f,ma = system.getdynamics()
pynamics.toc()
print('creating second order function...')
func = system.state_space_post_invert(f,ma,eq1_dd)
pynamics.toc()
#func = system.state_space_post_invert2(f,ma,eq1_dd,eq1_d,eq1)
#func = system.state_space_post_invert2(f,ma,eq1_dd,eq1_d,eq1,eq_active = b)
print('integrating...')
states=scipy.integrate.odeint(func,ini,t,rtol = error, atol = error, args=(alpha, beta),full_output = 1,mxstep = int(1e5))
states = states[0]
pynamics.toc()
print('calculating outputs..')
output = Output(outputlist,system)
output2 = Output([KE-PE],system)
y = output.calc(states)
y2 = output2.calc(states)
pynamics.toc()
#
plt.figure(0)
plt.plot(y[:,0:4:2],y[:,1:4:2],'ro')
plt.plot(y[:,4::2],y[:,5::2])
#plt.plot(t,y[:,1])
plt.axis('equal')
#
plt.figure(1)
plt.plot(t,y2[:])
#plt.axis('equal')
#

plt.figure(2)
plt.plot(y[-1,0::2],y[-1,1::2],'ro-')
plt.axis('equal')

plt.figure(3)
plt.plot(y[0,0::2],y[0,1::2],'ro-')
plt.axis('equal')
