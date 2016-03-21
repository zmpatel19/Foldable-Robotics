# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
from sympy import cos, sin
from pynamics.vector import Vector

class Rotation(object):
    def __init__(self,f1,f2,R,w_):
        self.f1 = f1
        self.f2 = f2
        self._R = R
        self.w_ = w_
    def to_other(self,f):
        if f==self.f1:
            return self._R
        elif f==self.f2:
            return self._R.T
        else:
            raise(Exception('frame not in this rotation'))
    def w__from(self,f):
        if f==self.f1:
            return self.w_
        elif f==self.f2:
            return -self.w_
        else:
            raise(Exception('frame not in this rotation'))
    def other(self,f):
        if f==self.f1:
            return self.f2
        elif f==self.f2:
            return self.f1
        else:
            raise(Exception('frame not in this rotation'))
     
class FixedAxisRotation(Rotation):
    def __init__(self,f1,f2,axis,q,sys):
        self.f1 = f1
        self.f2 = f2
        f1.add_rotation(self)
        f2.add_rotation(self)
#        self.q = q
#        self.axis = sympy.Matrix(axis)
#        self.w = 
#        w = q.diff().subs(sys.derivatives)
#        self.w_ = w*Vector({f1:axis})
        self._R,self.w_,self.fixedaxis = self.build_fixed_axis(axis,q,sys,f1)
        
    @staticmethod
    def build_fixed_axis(axis,q,sys,frame):
        axis = sympy.Matrix(axis)
        l_2 = axis.dot(axis)
        axis = axis/(l_2**.5)
        
        e0 = axis[0]        
        e1 = axis[1]        
        e2 = axis[2]        
        
        c = axis * axis.T
        a = (sympy.Matrix.eye(3) - c) * cos(q)
        b = sympy.Matrix([[0, -e2, e1], [e2, 0, -e0],[-e1, e0, 0]]) * sin(q)
        R = a + b + c 
        R = R.T

        w = sys.derivative(q)
        w_ = w*Vector({frame:axis})
        return R,w_,axis
