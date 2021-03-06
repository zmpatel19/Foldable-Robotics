# -*- coding: utf-8 -*-

"""Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license."""


#class FrameVector(object):
#    def __init__(self,frame,coords):
#        self.frame = frame
#        self.coords = sympy.Matrix(coords)
#    def __str__(self):
#        return str(self.frame)+str(self.coords[:])
#    def __repr__(self):
#        return str(self)

import sympy
sympy.init_printing(pretty_print=False)
import sys

ZERO = sympy.Number(0)
dimension = 3

import logging
logger = logging.getLogger('pynamics')
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#modulename = '__main__'

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler('pynamics.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

systemname = '_system'
integrator = 0
use_quaternions = False

def set_system(modulename,system):
    import pynamics
    pynamics.modulename = modulename
    module = sys.modules[modulename]
    setattr(module,systemname,system)
    
def get_system():
    import pynamics
    module = sys.modules[pynamics.modulename]
    return getattr(module,systemname)
            
import time
time0 = 0
def tic():
    global time0
    time0 = time.time()
def toc():
    logger.info(str(time.time()-time0))

#t = sympy.Symbol('t')

automatic_differentiate = True