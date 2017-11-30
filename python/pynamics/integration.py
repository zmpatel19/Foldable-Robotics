# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:36:33 2017

@author: daukes
"""

import numpy
import logging
logger = logging.getLogger('pynamics.integration')

def integrate_odeint(*arguments,**keyword_arguments):
    import scipy.integrate
    
    logger.info('beginning integration')
    result = scipy.integrate.odeint(*arguments,**keyword_arguments)
    logger.info('finished integration')
    return result
