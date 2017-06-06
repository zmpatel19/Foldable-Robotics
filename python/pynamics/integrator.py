# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:59:11 2017

@author: danaukes
using information from: http://lpsa.swarthmore.edu/NumInt/NumIntFourth.html
"""
import numpy

class RK4(object):

    def __init__(self,f,x0,t,tol = 1e-7,h_min = None):
        self.f = f
        self.t = t
        self.h = t[1]-t[0]
        self.tol = tol
        self.h_min = h_min or self.h*1e-3
        self.h_max = self.h
        self.x0 = x0
    def step(self,t,x,args = None):
        h = self.h
        f = self.f
        
        x = numpy.array(x)
        
        k1 = numpy.array(f(x,t,*args))
        k2 = numpy.array(f((x+k1*(h/2)),t+h/2,*args))
        k3 = numpy.array(f((x+k2*(h/2)),t+h/2,*args))
        k4 = numpy.array(f((x+k3*h),t+h,*args))        
        
        x1 = x+(k1+2*k2+2*k3+k4)/6*h
    
        return x1
    
    def run(self,args = None):
        x = []
        x.append(self.x0)
        for time in self.t:
            x.append(self.step(time,x[-1],args = args))
        return numpy.array(x)[1:]