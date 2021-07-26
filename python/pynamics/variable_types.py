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
    def __new__(self,name=None):
        
        name = name or self.generate_name()
        
        obj = sympy.Symbol.__new__(self,name)
        return obj

class Constant(sympy.Symbol,NameGenerator):
    def __new__(self,value=None,name=None,system = None):

        name = name or self.generate_name()

        system = system or pynamics.get_system()

        obj = sympy.Symbol.__new__(self,name)
#        obj.value = value
        system.add_constant(obj)
        if value is not None:
            system.add_constant_value(obj,value)
        return obj

class Differentiable(sympy.Symbol,NameGenerator):
    def __new__(cls,name=None,system = None,limit = 3,ii=0,ini = None,output_full=True):

        system = system or pynamics.get_system()

        name = name or cls.generate_name()

        
        output = []
        
        

        for kk,jj in enumerate(range(ii,limit)):
            

            if kk==0:
                subname = name
                variable = sympy.Symbol.__new__(cls,subname)
            else:
                subname = name+'_'+'d'*kk
                # if jj==limit-1:
                    # variable = Variable(subname)
                # else:
                    # variable = sympy.Symbol.__new__(cls,subname)
                variable = sympy.Symbol.__new__(cls,subname)

            output.append(variable)
            system.add_q(variable,jj)

        for item in output:
            item.set_derivative(sympy.Number(0))

        for kk,(a,a_d) in enumerate(zip(output[:-1],output[1:])):
            system.set_derivative(a,a_d)
            a.set_derivative(a_d)

            if ini is not None:
                system.set_ini(a,ini[kk])
        
        if output_full:
            if len(output)==1:
                return output[0]
            else:
                return output
        else:
            return output[0]
        # return output

    @property
    def _d(self):
        return self.get_derivative()
        
    def set_derivative(self,other):
        self._time_derivative=other

    def get_derivative(self):
        return self._time_derivative