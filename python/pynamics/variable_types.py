# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
import pynamics
from pynamics.name_generator import NameGenerator

class Variable(sympy.Symbol,NameGenerator):
    def __new__(self,name):
        
        name = name or self.generate_name()
        
        obj = sympy.Symbol.__new__(self,name)
        pynamics.addself(obj,name)
        return obj

class Constant(sympy.Symbol,NameGenerator):
    def __new__(self,name,value,system = None):

        name = name or self.generate_name()

        system = system or pynamics.get_system()

        obj = sympy.Symbol.__new__(self,name)
        obj.value = value
        system.add_constant(obj,value)
        pynamics.addself(obj,name)
        return obj

class Differentiable(sympy.Symbol,NameGenerator):
    def __new__(cls,sys = None,name=None,limit = 3,ii=0):

        sys = sys or pynamics.get_system()

        name = name or self.generate_name()

        differentiables = []

        for jj in range(ii,limit):
            

            if jj==0:
                subname = name
                variable = sympy.Symbol.__new__(cls,subname)
            else:
                subname = name+'_'+'d'*jj
                variable = sympy.Symbol.__new__(cls,subname)

            sys.add_q(variable,jj)
            differentiables.append(variable)
            pynamics.addself(variable,subname)

        for a,a_d in zip(differentiables[:-1],differentiables[1:]):
            sys.add_derivative(a,a_d)

        return differentiables 
