# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
import pynamics.variable_types
from pynamics.frame import Frame
from pynamics.system import System

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

system = System()
pynamics.set_system(__name__,system)
x,x_d,x_dd=pynamics.variable_types.Differentiable('x',system)
q1,q1_d,q1_dd=pynamics.variable_types.Differentiable('q1',system)

eq = x**2+2*x
eq_d = system.derivative(eq)

N = Frame('N',system)
A = Frame('A',system)

system.set_newtonian(N)
A.rotate_fixed_axis(N,[0,0,1],q1,system)


p1 = 3*A.x+2*N.y
v1=p1.time_derivative(N,system)
