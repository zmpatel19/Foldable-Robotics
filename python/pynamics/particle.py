# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics

class Particle(object):
    ii = 0
    def __init__(self,system,pCM,mass,name = None):
        if name==None:
            name = 'Particle{0:d}'.format(self.ii)
            type(self).ii+=1
        self.name = name
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
        
    def __repr__(self):
        return self.name+'(particle)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())
    def __str__(self):
        return self.name+'(particle)'
#        return self.name+' <frame {0:#x}>'.format(self.__hash__())