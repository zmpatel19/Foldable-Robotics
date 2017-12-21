# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import sympy

def solve_bayo_ledesma(system,f,ma,state,constants = None):
    constants = constants or {}
    state_d = [system.derivatives[item] for item in state]
    q_d = [item for item in state_d if item in system.get_q(1)]
    q_dd = [item for item in state_d if item in system.get_q(2)]
    Ax_b = sympy.Matrix(f)-sympy.Matrix(ma)
    Ax_b = Ax_b.subs(constants)
    A = Ax_b.jacobian(q_dd)
    b = -((Ax_b).subs(dict([(item,0) for item in q_dd])))
    return A,b