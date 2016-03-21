# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output

import sympy
import numpy
import scipy.integrate
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()


#e1 = Variable('e1')
#e2 = Variable('e2')
Ex = Variable('Ex')
Ey = Variable('Ey')
Ez = Variable('Ez')
F1x = Variable('F1x')
F1y = Variable('F1y')
F1z = Variable('F1z')
F2x = Variable('F2x')
F2y = Variable('F2y')
F2z = Variable('F2z')
x1 = Variable('x1')
y1 = Variable('y1')
z1 = Variable('z1')
x2 = Variable('x2')
y2 = Variable('y2')
z2 = Variable('z2')
xe = Variable('xe')
ye = Variable('ye')
ze = Variable('ze')

N = Frame('N')

F1 = F1x*N.x+F1y*N.y+F1z*N.z
F2 = F2x*N.x+F2y*N.y+F2z*N.z
E = Ex*N.x+Ey*N.y+Ez*N.z

r1 = x1*N.x + y1*N.y+z1*N.z
r2 = x2*N.x + y2*N.y+z2*N.z
re = xe*N.x + ye*N.y+ze*N.z

#o = 1*N.x+1*N.y+1*N.z
o = re
#o=0*N.x

M1 = (r1-o).cross(F1)
M2 = (r2-o).cross(F2)
Me = (re-o).cross(E)

Zf = F1+F2+E
Zm = M1+M2+Me

knowns = {}

knowns[x1]=356+90
knowns[y1]=0
knowns[z1]=0

knowns[x2]=-83-90
knowns[y2]=-155-100
knowns[z2]=0

knowns[xe]=0
knowns[ye]=0
knowns[ze]=140
#
knowns[Ex]=4899
knowns[Ey]=4899
knowns[Ez]=7936

Zfx = Zf.dot(N.x)
Zfy = Zf.dot(N.y)
Zfz = Zf.dot(N.z)

Zmx = Zm.dot(N.x)
Zmy = Zm.dot(N.y)
Zmz = Zm.dot(N.z)

zero = [Zfx,Zfy,Zfz,Zmx,Zmy,Zmz]
zero_m = sympy.Matrix(zero)
zero_m = zero_m.subs(knowns)
unknowns = sympy.Matrix(list(zero_m.atoms(Variable)))
#unknowns = sympy.Matrix([F1x,F1y,F1z,F2x,F2y,F2z,Ey,Ez])
A = zero_m.jacobian(unknowns)
b = A*unknowns-zero_m
#A = numpy.array(A.tolist(),float)
#b = numpy.array(b.tolist(),float)
#
#A_inv = scipy.linalg.inv(A)
#f = A_inv.dot(b)
#A_pinv = scipy.linalg.pinv(A)
#f = A_pinv.dot(b)
#
#f=f.flatten().tolist()
#solution = dict(zip(unknowns,f))
#sol = sympy.solve(zero_m,*unknowns)