# -*coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""
import pynamics
pynamics.script_mode = False
import logging
logger = logging.getLogger('pynamics_tests.init')
import subprocess
import pynamics_examples
import os
import sys
my_module = sys.modules['pynamics_examples']
my_dir = path = os.path.dirname(my_module.__file__)
s_template='jupyter nbconvert "{0}" --to python'

def convert_notebook(filename):
    path = os.path.join(my_dir,filename)
    s =s_template.format(path)
    print(s)
    subprocess.run(s,shell=True,capture_output=True)    

import pynamics_examples.babyboot
import pynamics_examples.ball_rope
import pynamics_examples.body_in_space
import pynamics_examples.body_in_space_local
import pynamics_examples.bouncy2
import pynamics_examples.bouncy_mod
import pynamics_examples.cart_pendulum
import pynamics_examples.cart_pendulum_forced_velocity
import pynamics_examples.differentiating
import pynamics_examples.falling_rod
# import pynamics_examples.five_bar_spherical
import pynamics_examples.four_bar
import pynamics_examples.glider
import pynamics_examples.motor_load_side
import pynamics_examples.motor_motor_side
import pynamics_examples.parallel_five_bar_jumper
import pynamics_examples.parallel_five_bar_jumper_foot
import pynamics_examples.pendulum_2_ways
import pynamics_examples.pendulum_chien_wen
import pynamics_examples.pendulum_in_water
import pynamics_examples.single_dof_bouncer
import pynamics_examples.springy_pendulum
import pynamics_examples.standing_stability_test
# import pynamics_examples.fitting.triple_pendulum_fitting
import pynamics_examples.triple_pendulum_inverse_dynamics
# import pynamics_examples.fitting.triple_pendulum_rft
import pynamics_examples.triple_pendulum_swimmer


# convert_notebook('triple_pendulum.ipynb')
# import pynamics_examples.triple_pendulum
# os.remove(my_dir,'triple_pendulum.py')
