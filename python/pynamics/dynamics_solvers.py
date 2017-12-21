# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 09:18:34 2017

@author: daukes
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