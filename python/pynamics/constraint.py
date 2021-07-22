# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:54:50 2020

@author: danaukes
"""

import sympy
import numpy

import logging
logger = logging.getLogger('pynamics.constraint')

class OutOfTol(Exception):
    pass

class KinematicConstraint(object):

    def __init__(self,eq):
        self.eq = eq
        # self.solved = False
        
    # def linearize(self,system):
    #     eq = self.eq
    #     eq_linear=[(system.derivative(item)) for item in eq]
    #     self.eq_linear = eq_linear

    # def solve(self,q_ind,q_dep,inv_method = 'LU'):
    #     self.q_dep = q_dep
    #     self.q_ind = q_ind
        
    #     logger.info('solving constraint')

    #     EQ = sympy.Matrix(self.eq_linear)
    #     AA = EQ.jacobian(sympy.Matrix(q_ind))
    #     BB = EQ.jacobian(sympy.Matrix(q_dep))
    
    #     CC = EQ - AA*(sympy.Matrix(q_ind)) - BB*(sympy.Matrix(q_dep))
    #     CC = sympy.simplify(CC)
    #     assert(sum(CC)==0)
    
    #     self.J = sympy.simplify(BB.solve(-(AA),method = inv_method))
        
    #     self.subs = dict([(a,b) for a,b in zip(q_dep,self.J*sympy.Matrix(q_ind))])
    #     self.solved = True
    #     return self.J, self.subs
    
    def solve_numeric(self,variables, guess, *my_constants,tol=1e-5):
        import scipy.optimize
        # all_substitutions = {}
        # print()
        # [all_substitutions.update(item) for item in substitutions]
        # print(all_substitutions)
        constants = {}
        for item in my_constants:
            constants.update(item)

        for item in variables:
            if item in constants:
                raise Warning('A variable which you are solving for is being supplied in constants',item)
                
        eq = [item.subs(constants) for item in self.eq]
        eq = numpy.array(eq)
        error = (eq**2).sum()
        f = sympy.lambdify(variables,error)
        def function(args):
            return f(*args)
        result = scipy.optimize.minimize(function,guess)
        if result.fun>tol:
            raise(OutOfTol())
        dict1 = dict([(key,value) for key,value in zip(variables,result.x)])
        return dict1

class AccelerationConstraint(object):

    def __init__(self,eq):
        self.eq = eq
        
    # def solve(self,inv_method = 'LU'):
        
    #     logger.info('solving constraint')

    #     EQ = sympy.Matrix(self.eq)
    #     AA = EQ.jacobian(sympy.Matrix(self.q_ind))
    #     BB = EQ.jacobian(sympy.Matrix(self.q_dep))
    
    #     CC = EQ - AA*(sympy.Matrix(self.q_ind)) - BB*(sympy.Matrix(self.q_dep))
    #     CC = sympy.simplify(CC)
    #     assert(sum(CC)==0)
    
    #     self.J = sympy.simplify(BB.solve(-(AA),method = inv_method))
        
    #     self.subs = dict([(a,b) for a,b in zip(self.q_dep,self.J*sympy.Matrix(self.q_ind))])
    #     self.solved = True
    #     return self.J, self.subs


    # def get_constraint_matrix(self,inv_method = 'LU'):
        
    #     logger.info('solving constraint')

    #     EQ = sympy.Matrix(self.eq)
    #     AA = EQ.jacobian(sympy.Matrix(self.q_ind))
    #     BB = EQ.jacobian(sympy.Matrix(self.q_dep))
    
    #     CC = EQ - AA*(sympy.Matrix(self.q_ind)) - BB*(sympy.Matrix(self.q_dep))
    #     CC = sympy.simplify(CC)
    #     assert(sum(CC)==0)
    
    #     self.J = sympy.simplify(BB.solve(-(AA),method = inv_method))
        
    #     return self.J
    
