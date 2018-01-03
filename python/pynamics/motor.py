# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
from pynamics.frame import Frame
from pynamics.dyadic import Dyadic
from pynamics.particle import PseudoParticle
from pynamics.body import Body

def add_motor_dynamics(B,Im,G,L,i,i_d,kt,kv,b,V,R,qB_d,axis,system):
    M = Frame('M')
    wNA = G*qB_d*axis
    aNA = wNA.time_derivative()
    I_motor = Dyadic.build(B,Im,Im,Im)
#    Motor = Body('Motor',B,0*M.x,0,I_motor,system,wNBody = wNA,alNBody = aNA)
    Motor = PseudoParticle(0*M.x,Im,name='Motor',system=system,vCM = wNA,aCM = aNA)
    Inductor = PseudoParticle(0*M.x,L,name='Inductor',system =system,vCM = i*M.x,aCM = i_d*M.x)
    T = kt*i
    system.addforce(T*axis,wNA)
    system.addforce(-b*wNA,wNA)
    system.addforce((V-i*R - kv*G*qB_d)*M.x,i*M.x)
    pass