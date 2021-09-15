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
        self.e = [e0,e1,e2,e3]

    @classmethod
    def build_from_axis_angle(cls,theta,x,y,z):
        e0 = cos(theta/2)
        s = sin(theta/2)
        e1 = s*x
        e2 = s*y
        e3 = s*z
        return UnitQuaternion(e0,e1,e2,e3)
        
    def norm(self):
        return self.norm_squared()**.5
    def norm_squared(self):
        e = sympy.Matrix(self.e)
        return sum([item**2 for item in e])
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
        e = []
        for item in self.e:
            if type(item) in [int,float]:
                a='{0:.3f}'.format(item)
            else:
                a=str(item)
            e.append(a)
        s = 'Q({0},{1},{2},{3})'.format(*e)
        return s
    def __repr__(self):
        return str(self)
    def hamilton_product(self,other):
        e01 = self.e[0]
        e02 = other.e[0]
        e11 = self.e[1]
        e12 = other.e[1]
        e21 = self.e[2]
        e22 = other.e[2]
        e31 = self.e[3]
        e32 = other.e[3]
        e0 = e01*e02-e11*e12-e21*e22-e31*e32
        e1 = e01*e12+e11*e02+e21*e32-e31*e22
        e2 = e01*e22-e11*e32+e21*e02+e31*e12
        e3 = e01*e32+e11*e22-e21*e12+e31*e02
        return Quaternion(e0,e1,e2,e3)
    def scalar(self):
        return self.e0
    def vector(self):
        vector = self.e[1:]
        return array(vector)
    def conjugate(self):
        new = type(self)(self.e[0],-self.e[1],-self.e[2],-self.e[3])
        return new
    def inv(self):
        new = self.conjugate()/self.norm_squared()
        return new
    def unit(self):
        result = self/self.norm()
        return UnitQuaternion(*result)
    def rotate_by(self,q):
        new = q.rotate(self)
        return new
    def sum(self,other):
        new = Quaternion(self.e[0]+other.e[0],self.e[1]+other.e[1],self.e[2]+other.e[2],self.e[3]+other.e[3])
        return new
    # def multiply(self,other):
    #     e0 = self.e0*other.e0-dot(self.vector(),other.vector())
    #     v = self.e0*other.vector()+other.e0*self.vector()+cross(self.vector(),other.vector())
    #     new = Quaternion(e0,*v)
    #     return new
    def expand(self):
        e = sympy.Matrix(self.e)
        new = Quaternion(*(e.expand()))
        return new
    def conjugation(self,other):
        result = other*self*other.inv()
        return result
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self.e[index]

        elif isinstance(index, slice):
            return self.e[index]

    def __setitem__(self, index, v):
        if isinstance(index, int):
            self.e[index] = v
            
        elif isinstance(index, slice):
            if isinstance(v,Quaternion):
                self.list[index] = v.e
            elif isinstance(v,list):
                self.list[index] = v
            else:
                raise(Exception())

    def __iter__(self):
        for item in self.e:
            yield item

    def __len__(self):
        return len(self.e)
    
    
    
class UnitQuaternion(Quaternion):
    @classmethod
    def build_from_axis_angle(cls,theta,x,y,z):
        e0 = cos(theta/2)
        s = sin(theta/2)
        e1 = s*x
        e2 = s*y
        e3 = s*z
        return cls(e0,e1,e2,e3)

    def hamilton_product(self, other):
        result = super(UnitQuaternion,self).hamilton_product(other)
        if isinstance(other,UnitQuaternion):
            result = UnitQuaternion(*result)
        return result



    def rotate(self,other):
        l=len(other)
        if l==3:
            other = Quaternion(0,*other)
        result = other.conjugation(self)
        if l==3:
            result=result.vector()
        if isinstance(other,UnitQuaternion):
            return UnitQuaternion(*result)
        else:
            return result
        # q = self
        # t = 2*cross(q.vector(),v.vector())
        # new = v.vector()+q.e[0]*t+cross(q.vector(),t)
        # new = Quaternion(sympy.Number(0),*new)
        # return new

    def inv(self):
        return self.conjugate()
    
    def unit(self):
        return self
    
class VectorQuaternion(Quaternion):
    def __init__(self,e1,e2,e3):
        self.e=[0,e1,e2,e3]
    
        
import sympy
a,b,c,d = sympy.symbols('a,b,c,d')    
e,f,g,h = sympy.symbols('e,f,g,h')    
q = sympy.Symbol('q')        

v1 = Quaternion(a,b,c,d) 
v12 = [b,c,d]
q = UnitQuaternion(e,f,g,h) 
# q = Quaternion.build_from_axis_angle(q, 0,0,1)
# v1 = Quaternion(0,2,3,4)
v2 = v1.rotate_by(q)
v22 = q*v1*q.inv()
v3 = q.rotate(v12)
