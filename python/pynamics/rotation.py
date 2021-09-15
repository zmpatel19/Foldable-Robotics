# -*- coding: utf-8 -*-

"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
from sympy import cos, sin
from pynamics.vector import Vector
import pynamics
from pynamics.quaternion import Quaternion,UnitQuaternion

def build_R_from_fixed_axis(axis,q,frame,sys):

    axis = sympy.Matrix(axis)
    l_2 = axis.dot(axis)
    axis = axis/(l_2**.5)
    
    e1 = axis[0]        
    e2 = axis[1]        
    e3 = axis[2]        
    
    c = axis * axis.T
    a = (sympy.Matrix.eye(3) - c) * cos(q)
    b = sympy.Matrix([[0, -e3, e2], [e3, 0, -e1],[-e2, e1, 0]]) * sin(q)
    R = a + b + c 
    R = R.T

    return R,axis

def w_from_fixed_axis(axis,q,frame,sys):

    axis = sympy.Matrix(axis)
    l_2 = axis.dot(axis)
    axis = axis/(l_2**.5)

    w = sys.derivative(q)
    w_ = w*Vector({frame:axis})
    return w_


class RotationBase(object):
    def __init__(self,f1,f2,r,q):
        self.f1 = f1
        self.f2 = f2
        self._r = r
        self._q = q
        
    def other(self,f):
        if f==self.f1:
            return self.f2
        elif f==self.f2:
            return self.f1
        else:
            raise(Exception('frame not in this rotation'))

class Rotation(RotationBase):
    def get_r_to(self,f):
        if f==self.f1:
            return self._r.T
        elif f==self.f2:
            return self._r
        else:
            raise(Exception('frame not in this rotation'))

    def get_rq_to(self,f):
        if f==self.f1:
            return self._q.inv()
        elif f==self.f2:
            return self._q
        else:
            raise(Exception('frame not in this rotation'))

    def get_r_from(self,f):
        return self.get_r_to(f).T

    def get_rq_from(self,f):
        return self.get_rq_to(f).inv()

    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))
        R,fixedaxis = build_R_from_fixed_axis(axis,q,f1,sys)
        q = UnitQuaternion.build_from_axis_angle(-q,*fixedaxis)
        new = cls(f1,f2,R,q)
        return new
    
class RotationalVelocity(RotationBase):
    def get_w_to(self,f):
        if f==self.f1:
            return -self._r
        elif f==self.f2:
            return self._r
        else:
            raise(Exception('frame not in this rotation'))

    def get_w_from(self,f):
        return -self.get_w_from(f)
            
    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))
        w = w_from_fixed_axis(axis,q,f1,sys)
        new = cls(f1,f2,w,Quaternion(0,0,0,0))
        return new

