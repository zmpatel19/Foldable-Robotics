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
from pynamics.quaternion import Quaternion

def build_fixed_axis(axis,q,frame,sys):

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

# def w_from_der(R,f,sys):
    
#     Rxx = R[0,0]
#     Rxy = R[0,1]
#     Rxz = R[0,2]

#     Ryx = R[1,0]
#     Ryy = R[1,1]
#     Ryz = R[1,2]

#     Rzx = R[2,0]
#     Rzy = R[2,1]
#     Rzz = R[2,2]
    
#     Rxx_d = sys.derivative(R[0,0])
#     Rxy_d = sys.derivative(R[0,1])
#     Rxz_d = sys.derivative(R[0,2])

#     Ryx_d = sys.derivative(R[1,0])
#     Ryy_d = sys.derivative(R[1,1])
#     Ryz_d = sys.derivative(R[1,2])

#     Rzx_d = sys.derivative(R[2,0])
#     Rzy_d = sys.derivative(R[2,1])
#     Rzz_d = sys.derivative(R[2,2])
    
#     wx = Rxz*Rxy_d + Ryz*Ryy_d + Rzz*Rzy_d
#     wy = Rxx*Rxz_d + Ryx*Ryz_d + Rzx*Rzz_d
#     wz = Rxy*Rxx_d + Ryy*Ryx_d + Rzy*Rzx_d
    
#     w = wx*f.x + wy*f.y + wz*f.z
#     return w


class RotationBase(object):
    def __init__(self,f1,f2,r):
        self.f1 = f1
        self.f2 = f2
        self._r = r
    def other(self,f):
        if f==self.f1:
            return self.f2
        elif f==self.f2:
            return self.f1
        else:
            raise(Exception('frame not in this rotation'))

class Rotation(RotationBase):
    def to_other(self,f):
        if f==self.f1:
            return self._r
        elif f==self.f2:
            return self._r.T
        else:
            raise(Exception('frame not in this rotation'))

    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))
        R,fixedaxis = build_fixed_axis(axis,q,f1,sys)
        new = cls(f1,f2,R)
        return new
    
class RotationalVelocity(RotationBase):
    def w__from(self,f):
        if f==self.f1:
            return self._r
        elif f==self.f2:
            return -self._r
        else:
            raise(Exception('frame not in this rotation'))
            
    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))
        w = w_from_fixed_axis(axis,q,f1,sys)
        new = cls(f1,f2,w)
        return new

class RotationQuaternion(RotationBase):
    def to_other(self,f):
        if f==self.f1:
            return self._r
        elif f==self.f2:
            return self._r.inv()
        else:
            raise(Exception('frame not in this rotation'))

    @classmethod
    def build(cls,e0,e1,e2,e3):
        q = Quaternion(e0,e1,e2,e3)
        new = cls(f1,f2,q)
        return new

    @classmethod
    def build_fixed_axis(cls,f1,f2,axis,q,sys):
        import pynamics.misc_tools
        if not all([pynamics.misc_tools.is_literal(item) for item in axis]):
            raise(Exception('not all axis variables are constant'))

        axis = sympy.Matrix(axis)
        l_2 = axis.dot(axis)
        axis = axis/(l_2**.5)
        axis *= sin(q)
        
        e1 = axis[0]        
        e2 = axis[1]        
        e3 = axis[2]        
        
        e0 = cos(q)
        q = Quaternion(e0,e1,e2,e3)
        new = cls(f1,f2,q)
        return new

class RotationQuaternion_d(RotationBase):
    pass
