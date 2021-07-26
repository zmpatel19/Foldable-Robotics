# -*- coding: utf-8 -*-
"""
Created on Tue May 25 10:24:05 2021

@author: danaukes
https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions
https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
"""

import sympy
sympy.init_printing(pretty_print=False)
from sympy import sin,cos,tan,pi,acos
import numpy

def array(input1):
    # return numpy.array(input1)
    return sympy.Matrix(input1)

def cross(a,b):
    # return numpy.cross(a,b)
    return a.cross(b)

def dot(a,b):
    return a.dot(b)

class Quaternion(object):
    
    def __init__(self,e0,e1,e2,e3):
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    @classmethod
    def build_from_axis_angle(cls,theta,x,y,z):
        e0 = cos(theta/2)
        s = sin(theta/2)
        e1 = s*x
        e2 = s*y
        e3 = s*z
        return UnitQuaternion(e0,e1,e2,e3)
        
    # def p(self):
        # return self
    # def inv(self):
        # return self.conjugate()
    def norm(self):
        return self.norm_squared()**.5
    def norm_squared(self):
        return (sum([self.e0**2,self.e1**2,self.e2**2,self.e3**2]))
    def __mul__(self,other):
        if type(other) in [int,float]:
            other = Quaternion(other,0,0,0)
        return self.hamilton_product(other)
    def __truediv__(self,other):
        if type(other) in [int,float]:
            other = Quaternion(1/other,0,0,0)
            return self.hamilton_product(other)
        else:
            other = Quaternion(1/other,0,0,0)
            return self.hamilton_product(other)
            # raise TypeError
    def __str__(self):
        if type(self.e0) in [int,float]:
            e0='{0:.3f}'.format(self.e0)
        else:
            e0=str(self.e0)
        if type(self.e1) in [int,float]:
            e1='{0:.3f}'.format(self.e1)
        else:
            e1=str(self.e1)
        if type(self.e2) in [int,float]:
            e2='{0:.3f}'.format(self.e2)
        else:
            e2=str(self.e2)
        if type(self.e3) in [int,float]:
            e3='{0:.3f}'.format(self.e3)
        else:
            e3=str(self.e3)
        s = 'Q({0},{1},{2},{3})'.format(e0,e1,e2,e3)
        return s
    def __repr__(self):
        return str(self)
    def hamilton_product(self,other):
        e01 = self.e0
        e02 = other.e0
        e11 = self.e1
        e12 = other.e1
        e21 = self.e2
        e22 = other.e2
        e31 = self.e3
        e32 = other.e3
        e0 = e01*e02-e11*e12-e21*e22-e31*e32
        e1 = e01*e12+e11*e02+e21*e32-e31*e22
        e2 = e01*e22-e11*e32+e21*e02+e31*e12
        e3 = e01*e32+e11*e22-e21*e12+e31*e02
        return Quaternion(e0,e1,e2,e3)
    def scalar(self):
        return self.e0
    def vector(self):
        vector = self.e1,self.e2,self.e3  
        return array(vector)
    def conjugate(self):
        new = type(self)(self.e0,-self.e1,-self.e2,-self.e3)
        return new
    def inv(self):
        new = self.conjugate()/self.norm_squared()
        return new
    def unit(self):
        return self/self.norm()
    def rotate_by(self,q):
        new = q.rotate(self)
        return new
    def sum(self,other):
        new = Quaternion(self.e0+other.e0,self.e1+other.e1,self.e2+other.e2,self.e3+other.e3)
        return new
    def multiply(self,other):
        e0 = self.e0*other.e0-dot(self.vector(),other.vector())
        v = self.e0*other.vector()+other.e0*self.vector()+cross(self.vector(),other.vector())
        new = Quaternion(e0,*v)
        return new
    def expand(self):
        new = Quaternion(self.e0.expand(),self.e1.expand(),self.e2.expand(),self.e3.expand())
        return new
    
    
class UnitQuaternion(Quaternion):
    @classmethod
    def build_from_axis_angle(cls,theta,x,y,z):
        e0 = cos(theta/2)
        s = sin(theta/2)
        e1 = s*x
        e2 = s*y
        e3 = s*z
        return cls(e0,e1,e2,e3)

    def rotate(self,v):
        q = self
        t = 2*cross(q.vector(),v.vector())
        new = v.vector()+q.e0*t+cross(q.vector(),t)
        new = Quaternion(sympy.Number(0),*new)
        return new

    def inv(self):
        new = self.conjugate()
        return new

    def unit(self):
        return self
        
import sympy
a,b,c,d = sympy.symbols('a,b,c,d')    
e,f,g,h = sympy.symbols('e,f,g,h')    
q = sympy.Symbol('q')        

v1 = Quaternion(a,b,c,d) 
q = UnitQuaternion(e,f,g,h) 
# q = Quaternion.build_from_axis_angle(q, 0,0,1)
# v1 = Quaternion(0,2,3,4)
v2 = v1.rotate_by(q)
v22 = q*v1*q.inv()
v222 = q*v1*q.conjugate()

print(v2)