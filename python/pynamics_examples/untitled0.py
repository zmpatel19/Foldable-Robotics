# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 14:22:12 2022

@author: dongting
"""

import numpy
import socket
import time
import os
import datetime
import matplotlib.pyplot as plt
import math
import yaml
import logging
import time
import matplotlib
import matplotlib.pyplot as plt


from scipy.signal import butter,filtfilt

force_data_loc = numpy.genfromtxt('stiffness.csv')
force_data_loc[:,-4:-1] = force_data_loc[:,-4:-1]-force_data_loc[0,-4:-1]
# Filter requirements.
T = force_data_loc[-1,0]-force_data_loc[0,0]     # Sample Period

fs = len(force_data_loc)/T     # sample rate, Hz
cutoff = 0.5   # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 2       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

def butter_lowpass_filter(data, cutoff, fs, order):
    from scipy.signal import butter,filtfilt
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
y = butter_lowpass_filter(force_data_loc[:,3], cutoff, fs, order)

dis = force_data_loc[:,-2]*numpy.sqrt(2)*1000

plt.figure()
plt.plot(dis,force_data_loc[:,3],'b',alpha=0.2)
plt.plot(dis,y,'b-')
plt.rcParams["font.family"] = "Times New Roman"
plt.xlabel("Distance (mm)")
plt.ylabel("Force (N)")
plt.show()
# plt.vlines(60,-8,0.5)
