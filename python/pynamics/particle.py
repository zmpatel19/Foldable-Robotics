# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics

class Particle(object):
    _ii = 0
    typestring = 'Particle'
    def __init__(self,pCM,mass,name = None,system = None):
        system = system or pynamics.get_system()

        self.name = name or self.generate_name()
        self.pCM = pCM
        self.mass = mass
        self.system = system

        self.vCM=self.pCM.diff_in_parts(self.system.newtonian,self.system)
        self.aCM=self.vCM.diff_in_parts(self.system.newtonian,self.system)
                
        self.effectiveforce = self.mass*self.aCM
        self.KE = .5*mass*self.vCM.dot(self.vCM)
#        self.linearmomentum = self.mass*self.vCM
        
        self.system.particles.append(self)
        self.adddynamics()

    def adddynamics(self):
        self.system.addeffectiveforce(self.effectiveforce,self.vCM)
#        self.system.addmomentum(self.linearmomentum,self.vCM)
        self.system.addKE(self.KE)
        
    def addforcegravity(self,gravityvector):
        self.gravityvector = gravityvector
        self.forcegravity = self.mass*gravityvector
        self.system.addforce(self.forcegravity,self.vCM)
        
    def generate_name(self):
        name = '{0}{1:04d}'.format(self.typestring,self._ii)
        type(self)._ii+=1
        return name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
