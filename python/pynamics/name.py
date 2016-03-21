# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

class Name(object):
    _ii = 0
    @classmethod
    def getname(cls):
        cls._ii+=1
        return 'dummy'+str(cls._ii)
    @classmethod
    def point(cls):
        cls._ii+=1
        return 'pt{0:04.0f}'.format(cls._ii)
    @classmethod
    def particle(cls):
        cls._ii+=1
        return 'part{0:04.0f}'.format(cls._ii)
    @classmethod
    def frame(cls):
        cls._ii+=1
        return 'frame{0:04.0f}'.format(cls._ii)