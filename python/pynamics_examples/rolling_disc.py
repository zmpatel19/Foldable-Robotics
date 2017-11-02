# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
pynamics.script_mode = False
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

#import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

error = 1e-12

tinitial = 0
tfinal = 10
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]


m = Constant(1,'m',system)
g = Constant(9.81,'g',system)
I = Constant(1,'I',system)
J = Constant(1,'J',system)
r = Constant(1,'L',system)

H,H_d,H_dd = Differentiable('H',system)
L,L_d,L_dd = Differentiable('L',system)
Q,Q_d,Q_dd = Differentiable('Q',system)
x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)

initialvalues = {}
initialvalues[H]=0*pi/180
initialvalues[H_d]=0*pi/180
initialvalues[L]=0*pi/180
initialvalues[L_d]=0*pi/180
initialvalues[Q]=0*pi/180
initialvalues[Q_d]=0*pi/180
initialvalues[x]=0
initialvalues[x_d]=0
initialvalues[y]=0
initialvalues[y_d]=0

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

A = Frame('A')
B = Frame('B')
C = Frame('C')
D = Frame('D')

system.set_newtonian(A)
B.rotate_fixed_axis_directed(A,[0,0,1],H,system)
C.rotate_fixed_axis_directed(B,[1,0,0],-L,system)
D.rotate_fixed_axis_directed(C,[0,1,0],Q,system)

pNA=0*A.x
pDA = pNA+x*A.x+y*A.y
pBcm = pDA+r*C.z

wAD = A.getw_(D)

II = Dyadic.build(B,J,I,J)

BodyD = Body('BodyD',D,pBcm,m,II,system)

#ParticleA = Particle(pAcm,mA,'ParticleA',system)
#ParticleB = Particle(pBcm,mB,'ParticleB',system)
#ParticleC = Particle(pCcm,mC,'ParticleC',system)

system.addforcegravity(-g*A.z)

f,ma = system.getdynamics()
