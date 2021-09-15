# -*coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import pynamics
import logging
logger = logging.getLogger('pynamics_tests.init')
import subprocess
import pynamics_examples
import os
import sys
import importlib
import glob
import inspect

my_module = sys.modules['pynamics_examples']
my_dir = path = os.path.dirname(my_module.__file__)
s_template='jupyter nbconvert "{0}" --to python'

def convert_notebook(filename):
    path = os.path.join(my_dir,filename)
    s =s_template.format(path)
    print(s)
    subprocess.run(s,shell=True,capture_output=True)    


m = sys.modules[__name__]

f = inspect.getfile(my_module)
print(f)
d = os.path.split(f)[0]

p = os.path.join(d,'*.py')
files = glob.glob(p)
print(files)
if f in files:
    files.pop(files.index((f)))
print(files)

for item in files:
    item = os.path.split(item)[1]
    modulename = item.split('.py')[0]
    importlib.import_module(modulename)


convert_notebook('triple_pendulum.ipynb')
import pynamics_examples.triple_pendulum
os.remove(os.path.join(my_dir,'triple_pendulum.py'))
