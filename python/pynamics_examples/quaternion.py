#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:27:42 2021

@author: danaukes
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
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi


system = System()
pynamics.set_system(__name__,system)


e0,e0_d,e0_dd = Differentiable('e0')
e1,e1_d,e1_dd = Differentiable('e1')
e2,e2_d,e2_dd = Differentiable('e2')
e3,e3_d,e3_dd = Differentiable('e3')

qA,qA_d,qA_dd = Differentiable('qA')
qB,qB_d,qB_dd = Differentiable('qB')
qC,qC_d,qC_dd = Differentiable('qC')


N = Frame('N',system)
A = Frame('A',system)
B = Frame('B',system)

v = e1*N.x+e2*N.y+e3*N.z

eq = e0**2+e1**2++e2**2+e3**2-1