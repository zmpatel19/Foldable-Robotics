# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import numpy
import pynamics

class Output(object):
    def __init__(self,y_exp, system=None, constant_values = None):
        import sympy
        system = system or pynamics.get_system()

        constant_values = constant_values or system.constant_values
        self.y_expression = sympy.Matrix(y_exp)
        cons_s = list(constant_values.keys())
        self.cons_v = [constant_values[key] for key in cons_s]
        self.fy_expression = sympy.lambdify(system.get_state_variables()+cons_s,self.y_expression)

    def calc(self,x):
        self.y = numpy.array([self.fy_expression(*(item.tolist()+self.cons_v)) for item in x]).squeeze()
        return self.y

    def plot_time(self,t=None):
        import matplotlib.pyplot as plt
        plt.figure()
        try:
            self.y
        except AttributeError:
            self.calc()
        if t is None:
            plt.plot(self.y)
        else:
            plt.plot(t,self.y)

#class PointsOutput(Output):
#    def init(self,y_exp, system=None, constant_values = None):
#        system = system or pynamics.get_system()
#        y_exp = [item for item2 in y_exp for item in [item2.dot(system.newtonian.x),item2.dot(system.newtonian.y)]]
#        super(PointsOutput,self).__init__(y_exp, system, constant_values )
#
#    def plot_xy(self):
#        def plot_time(self,t=None):
#        import matplotlib.pyplot as plt
#        plt.figure()
#        try:
#            self.y
#        except AttributeError:
#            self.calc()
#        if t is None:
#            plt.plot(self.y)
#        else:
#            plt.plot(t,self.y)
        