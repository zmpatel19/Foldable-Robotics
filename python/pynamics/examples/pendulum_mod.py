# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
#pynamics.script_mode = True
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant,Variable
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output
from pynamics.particle import Particle

import sympy
import numpy
import scipy.integrate
import matplotlib.pyplot as plt
plt.ion()
from sympy import pi
system = System()

def der(eq,system):
    b = sympy.Number(0)
    for item in system.get_q(0)+system.get_q(1):
        b+=eq.diff(item)*system.derivative(item)
    return b

lA = Constant('lA',1,system)

mA = Constant('mA',1,system)

g = Constant('g',9.81,system)
b = Constant('b',1e0,system)
k = Constant('k',1e1,system)

tinitial = 0
tfinal = 5
tstep = .001
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant('preload1',0*pi/180,system)

x,x_d,x_dd = Differentiable(system,'x')
y,y_d,y_dd = Differentiable(system,'y')
f = Variable('f')

initialvalues = {}
initialvalues[x]=1
initialvalues[x_d]=0
initialvalues[y]=0
initialvalues[y_d]=0

statevariables = system.get_q(0)+system.get_q(1)
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')

system.set_newtonian(N)

pNA=0*N.x
pAB=pNA+x*N.x+y*N.y
vAB=pAB.time_derivative(N,system)

ParticleA = Particle(system,pAB,mA,'ParticleA')

system.addforce(-b*vAB,vAB)

system.addforcegravity(-g*N.y)

x1 = ParticleA.pCM.dot(N.x)
y1 = ParticleA.pCM.dot(N.y)
KE = system.KE
PE = system.getPEGravity(pNA) - system.getPESprings()

pynamics.tic()
print('solving dynamics...')
f,ma = system.getdynamics()
print('creating second order function...')

v = pAB-pNA
u = (v.dot(v))**.5

eq = [(v.dot(v)) - lA**2]
b=der(der(eq[0],system),system)
b = sympy.Matrix([b])
q2 = sympy.Matrix(system.get_q(2))
J = b.jacobian(q2)
c = (b-J*q2).expand()

#system.addforce(-f*u,vAB)

statevariables = system.get_q(0)+system.get_q(1)
augmented = [f]

def createsecondorderfunction2(system,f,ma,J,c):
    q_state = system.get_q(0)+system.get_q(1)
#    q_state_d = system.get_q(1) + system.get_q(2)
    f = sympy.Matrix(f)
    ma = sympy.Matrix(ma)
    q = system.get_q(0)
    q_d = system.get_q(1)
    q_dd = system.get_q(2)

    Ax_b = ma-f
    x = sympy.Matrix(q_dd)
    A = Ax_b.jacobian(x)
    b = -Ax_b.subs(dict(list([(item,0) for item in x])))
    
    m = len(q_d)
    n = J.shape[0]
    A_full = sympy.zeros(m+n)   
    A_full[:m,:m] = A
    A_full[m,:m] = J
    A_full[:m,m] = J.T

    b_full = sympy.zeros(m+n,1)
    b_full[:m,0]=b
    b_full[m:,0]=-c
    
    c_sym = list(system.constants.keys())
    c_val = [system.constants[key] for key in c_sym]

    fA = sympy.lambdify(q_state+c_sym,A_full)
    fb = sympy.lambdify(q_state+c_sym,b_full)

    def func(state,time):
        a = list(state)+c_val
        Ai = fA(*a)
        bi = fb(*a)
        x1 = state[m:]
        x2 = numpy.array(scipy.linalg.inv(Ai).dot(bi)).flatten()
        x3 = numpy.r_[x1,x2[:m]]
        x4 = x3.flatten().tolist()
        return x4
    return func

func1 = createsecondorderfunction2(system,f,ma,J,c)
print('integrating...')
states=scipy.integrate.odeint(func1,ini,t,rtol=1e-12,atol=1e-12,hmin=1e-14)
pynamics.toc()
print('calculating outputs..')
output = Output([x1,y1,KE-PE,x,y],system)
y = output.calc(states)
pynamics.toc()

plt.figure(1)
plt.hold(True)
plt.plot(y[:,0],y[:,1])
plt.axis('equal')

plt.figure(2)
plt.plot(y[:,2])

plt.figure(3)
plt.hold(True)
plt.plot(t,y[:,3])
plt.show()
