# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.dyadic import Dyadic

import sympy


if __name__=='__main__':
    from sympy import sin, cos
    from math import pi
    sys = System()

    N = Frame('N')
    A = Frame('A')
    B = Frame('B')
    C = Frame('C')
    D = Frame('D')
    E = Frame('E')

    a = Constant('a',4,sys)
#    a = sympy.Symbol('a')    
#    a,b,c,d,e,f,q = sympy.symbols('a b c d e f q')
    qA,qA_d,qA_dd = Differentiable('qA',sys)
    qB,qB_d,qB_dd = Differentiable('qB',sys)
    qC,qC_d,qC_dd = Differentiable('qC',sys)
    qD,qD_d,qD_dd = Differentiable('qD',sys)
    qE,qE_d,qE_dd = Differentiable('qE',sys)

    sys.set_newtonian(N)
    A.rotate_fixed_axis_directed(N,[0,0,1],qA,sys)
    B.rotate_fixed_axis_directed(A,[0,0,1],qB,sys)
    C.rotate_fixed_axis_directed(B,[0,0,1],qC,sys)
    D.rotate_fixed_axis_directed(C,[0,0,1],qD,sys)
    E.rotate_fixed_axis_directed(D,[1,0,0],qE,sys)
    
    a= N.efficient_rep(B,'dot')[frozenset((N.x_sym,B.x_sym))]