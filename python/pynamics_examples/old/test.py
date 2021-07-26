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
from pynamics.dyadic import Dyadic

import sympy


if __name__=='__main__':
    from sympy import sin, cos
    from math import pi
    sys = System()

    N = Frame('N',system)
    A = Frame('A',system)
    B = Frame('B',system)
    C = Frame('C',system)
    D = Frame('D',system)
    E = Frame('E',system)

    a = Constant('a',4,sys)
#    a = sympy.Symbol('a')    
#    a,b,c,d,e,f,q = sympy.symbols('a b c d e f q')
    qA,qA_d,qA_dd = Differentiable('qA',sys)
    qB,qB_d,qB_dd = Differentiable('qB',sys)
    qC,qC_d,qC_dd = Differentiable('qC',sys)
    qD,qD_d,qD_dd = Differentiable('qD',sys)
    qE,qE_d,qE_dd = Differentiable('qE',sys)

    sys.set_newtonian(N)
    A.rotate_fixed_axis(N,[0,0,1],qA,sys)
    B.rotate_fixed_axis(A,[0,0,1],qB,sys)
    C.rotate_fixed_axis(B,[0,0,1],qC,sys)
    D.rotate_fixed_axis(C,[0,0,1],qD,sys)
    E.rotate_fixed_axis(D,[1,0,0],qE,sys)
    
    a= N.efficient_rep(B,'dot')[frozenset((N.x_sym,B.x_sym))]