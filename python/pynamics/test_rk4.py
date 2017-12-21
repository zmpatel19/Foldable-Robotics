# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import numpy as np 
import matplotlib.pyplot as plt
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from rk4 import rkf, F

x0 = np.array([1, 0], dtype=np.float64)
f = F(a=0.1)

t = np.linspace(0, 30, 100)
y = rkf(f, t, x0)

plt.plot(t, y)
plt.show()