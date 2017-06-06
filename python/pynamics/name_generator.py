# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 17:35:06 2017

@author: danaukes
"""

class NameGenerator(object):

    def generate_name(self):
        try:
            self._generate_name()
        except AttributeError:
            type(self)._ii = 0
            return self._generate_name()
        return name
    
    def _generate_name(self):
        name = '{0}{1:04d}'.format(self.typestring,self._ii)
        type(self)._ii+=1
        return name

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)
    