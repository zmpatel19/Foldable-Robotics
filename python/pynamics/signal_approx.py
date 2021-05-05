# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:01:40 2021

@author: danaukes
"""
import sympy
sympy.init_session(use_latex=False,quiet=True)
# from math import pi
from sympy import pi
from sympy import sin,cos,acos,atan
arctan = lambda x: atan(x)
arccos = lambda x: acos(x)
# from numpy import sin,cos,arccos,arctan
# import numpy.arccos as acos
import matplotlib.pyplot as plt

x = sympy.Symbol('x')
d = sympy.Symbol('d')

trg = 1 - 2*arccos((1 - d)*sin(2*pi*x))/pi
sqr = 2*arctan(sin(2*pi*x)/d)/pi

f_trg = lambda x,d:(1 - 2*arccos((1 - d)*sin(2*pi*x))/pi)
f_sqr = lambda x,d: (2*arctan(sin(2*pi*x)/d)/pi)

swt = ((1 + f_trg((2*x - 1)/4,d)*f_sqr(x/2,d))/2)
f_swt = lambda x,d: ((1 + f_trg((2*x - 1)/4,d)*f_sqr(x/2,d))/2)

if __name__=='__main__':
    import numpy
    x_num = numpy.r_[-2:2:.01]
    d_num = .01

    f_trg2 = sympy.lambdify((x,d),trg)
    f_sqr2 = sympy.lambdify((x,d),sqr)
    f_swt2 = sympy.lambdify((x,d),swt)
    
    plt.plot(x_num,f_trg2(x_num,d_num))
    plt.plot(x_num,f_sqr2(x_num,d_num))
    plt.plot(x_num,f_swt2(x_num,d_num))
    
