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
import sympy
import scipy
import logging
pynamics.logger.setLevel(logging.ERROR)
pynamics.system.logger.setLevel(logging.ERROR)

import cma

import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi, sin, cos
system = System()
pynamics.set_system(__name__,system)


tinitial = 0
tfinal = 5
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

ang_ini = 0

v=1

x,x_d,x_dd = Differentiable('x',ini=[0,v*cos(ang_ini*pi/180)])
y,y_d,y_dd = Differentiable('y',ini=[1,v*sin(ang_ini*pi/180)])
z,z_d,z_dd = Differentiable('z',ini=[0,0])

qA,qA_d,qA_dd = Differentiable('qA',ini=[0,0])
qB,qB_d,qB_dd = Differentiable('qB',ini=[0,0])
qC,qC_d,qC_dd = Differentiable('qC',ini=[ang_ini*pi/180,0])


# mC = Constant(0,'mC')
g = Constant(9.81,'g')
I_11=Constant(6e-3,'I_11')
rho = Constant(1.292,'rho')
r = Constant(0,'r')
# Sw = Constant(.1,'Sw')
# Se = Constant(.025,'Se')
l = Constant(.35,'l')
lw = Constant(-.03,'lw')
le = Constant(.04,'le')
qE = Constant(3*pi/180,'qE')

ini = system.get_ini()

N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)
C = Frame('C',system)
E = Frame('E',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[1,0,0],qA,system)
B.rotate_fixed_axis(A,[0,1,0],qB,system)
C.rotate_fixed_axis(B,[0,0,1],qC,system)
E.rotate_fixed_axis(C,[0,0,1],-qE,system)

pCcm=x*N.x+y*N.y+z*N.z
pCcp=pCcm-lw*C.x

mC = pi*r**2*.1

pC1 = pCcm
pC2 = pCcm-l*C.x
pE = pC2-le*E.x

vcm = pCcm.time_derivative()

IC = Dyadic.build(C,I_11,I_11,I_11)

Body('BodyC',C,pCcm,mC,IC)

Area = 2*pi*r**2

vcp=pCcp.time_derivative()
f_aero_C = rho * vcp.length()*(vcp.dot(C.y))*Area*C.y

ve=pE.time_derivative()
f_aero_E = rho * ve.length()*(ve.dot(E.y))*Area*E.y


system.addforcegravity(-g*N.y)
system.addforce(-f_aero_C,vcp)
system.addforce(-f_aero_E,ve)

points = [pC1,pC2]

f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma)

def run(args):
    my_r = args[0]
    constants = system.constant_values.copy()
    constants[r] = my_r
    
    states=pynamics.integration.integrate_odeint(func1,ini,t, args=({'constants':constants},))
        
    return states

def measure_perf(args):
    print('r: ',args[0])
    if args[0]>1:
        return 1000
    if args[0]<=0:
        return 1000
    try:
        states = run(args)
        perf = 1/states[-1,0]
        return perf
    except scipy.linalg.LinAlgError:
        return 1000

yy = []    
xx = numpy.r_[0.1:1:5j]
for ii in xx:
    yy.append(measure_perf([ii]))
    
yy = numpy.array(yy)
plt.plot(xx,yy)

states = run([.1])
po = PointsOutput(points,system)
po.calc(states,t)
po.plot_time()
# po.animate(fps = 30, movie_name='glider.mp4',lw=2,marker='o')
