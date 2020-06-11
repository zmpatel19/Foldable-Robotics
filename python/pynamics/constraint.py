# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:54:50 2020

@author: danaukes
"""

import sympy

import logging
logger = logging.getLogger('pynamics.constraint')

class Constraint(object):

    def __init__(self,eq,q_ind,q_dep):
        self.eq = eq
        self.q_ind = q_ind
        self.q_dep = q_dep
        self.solved = False

    def solve(self,inv_method = 'LU'):
        # eq = sympy.Matrix(eq)
        
        
        logger.info('solving constraint')

        EQ = sympy.Matrix(self.eq)
        AA = EQ.jacobian(sympy.Matrix(self.q_ind))
        BB = EQ.jacobian(sympy.Matrix(self.q_dep))
    
        CC = EQ - AA*(sympy.Matrix(self.q_ind)) - BB*(sympy.Matrix(self.q_dep))
        CC = sympy.simplify(CC)
        assert(sum(CC)==0)
    
        self.J = sympy.simplify(BB.solve(-(AA),method = inv_method))
        
        self.subs = dict([(a,b) for a,b in zip(self.q_dep,self.J*sympy.Matrix(self.q_ind))])
        self.solved = True
        return self.J, self.subs
        
        