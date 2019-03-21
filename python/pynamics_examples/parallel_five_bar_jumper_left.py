# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import pynamics
from pynamics.frame import Frame
from pynamics.variable_types import Differentiable,Constant
from pynamics.system import System
from pynamics.body import Body
from pynamics.dyadic import Dyadic
from pynamics.output import Output,PointsOutput
from pynamics.particle import Particle
import pynamics.integration

import sympy
import numpy
import matplotlib.pyplot as plt
plt.ion()
from math import pi
system = System()
tol=1e-9


lO = Constant(.5,'lO',system)
lA = Constant(.5,'lA',system)
lB = Constant(1,'lB',system)
lC = Constant(.5,'lC',system)
lD = Constant(1,'lD',system)

mO = Constant(10,'mO',system)
mA = Constant(.1,'mA',system)
mB = Constant(.1,'mB',system)
mC = Constant(.1,'mC',system)
mD = Constant(.1,'mD',system)

I_main = Constant(1,'I_main',system)

g = Constant(9.81,'g',system)
b = Constant(1e0,'b',system)
k = Constant(1e1,'k',system)
stall_torque = Constant(1e2,'stall_torque',system)

k_constraint = Constant(1e5,'k_constraint',system)
b_constraint = Constant(1e3,'b_constraint',system)

tinitial = 0
tfinal = 10
tstep = 1/30
t = numpy.r_[tinitial:tfinal:tstep]

preload1 = Constant(0*pi/180,'preload1',system)
preload2 = Constant(0*pi/180,'preload2',system)
preload3 = Constant(-180*pi/180,'preload3',system)
preload4 = Constant(0*pi/180,'preload4',system)
preload5 = Constant(180*pi/180,'preload5',system)

x,x_d,x_dd = Differentiable('x',system)
y,y_d,y_dd = Differentiable('y',system)
qO,qO_d,qO_dd = Differentiable('qO',system)
#qA,qA_d,qA_dd = Differentiable('qA',system)
#qB,qB_d,qB_dd = Differentiable('qB',system)
qC,qC_d,qC_dd = Differentiable('qC',system)
qD,qD_d,qD_dd = Differentiable('qD',system)

initialvalues={
        x: 0,
        x_d: 0,
        y: 2,
        y_d: 0,
        qO: 0,
        qO_d: 0,
#        qA: -0.89,
#        qA_d: 0,
#        qB: -2.64,
#        qB_d: 0,
        qC: -pi+0.89,
        qC_d: 0,
        qD: -pi+2.64,
        qD_d: 0
        }

statevariables = system.get_state_variables()
ini = [initialvalues[item] for item in statevariables]

N = Frame('N')
O = Frame('O')
#A = Frame('A')
#B = Frame('B')
C = Frame('C')
D = Frame('D')

system.set_newtonian(N)
O.rotate_fixed_axis_directed(N,[0,0,1],qO,system)
#A.rotate_fixed_axis_directed(N,[0,0,1],qA,system)
#B.rotate_fixed_axis_directed(N,[0,0,1],qB,system)
C.rotate_fixed_axis_directed(N,[0,0,1],qC,system)
D.rotate_fixed_axis_directed(N,[0,0,1],qD,system)

pOrigin = 0*N.x+0*N.y
pOcm=x*N.x+y*N.y
pOA = pOcm+lO/2*O.x
pOC = pOcm-lO/2*O.x
#pAB = pOA+lA*A.x
#pBtip = pAB + lB*B.x
#vBtip = pBtip.time_derivative(N,system)

pCD = pOC + lC*C.x
pDtip = pCD + lD*D.x
vDtip = pDtip.time_derivative(N,system)

#def gen_init():
#    eqs = []
#    eqs.append(pBtip-pDtip)
##    eqs.append(pBtip-pOrigin)
#    a=[(item).express(N) for item in eqs]
#    b=[item.subs(system.constant_values) for item in a]
#    c = numpy.array([vec.dot(item) for vec in b for item in list(N.principal_axes)])
#    d = (c**2).sum()
#    e = system.get_state_variables()
#    #e = sorted(list(d.atoms(Differentiable)),key=lambda x:str(x))
#    f = sympy.lambdify(e,d)
#    g = lambda args:f(*args)
#    return g
#fun = gen_init()

#ini1=[initialvalues[item] for item in e]

#import scipy.optimize
#result = scipy.optimize.minimize(fun,ini)

#if result.fun<1e-7:
#    points = [pDtip,pCD,pOC,pOA,pAB,pBtip]
#    points = PointsOutput(points)
#    state = numpy.array([ini,result.x])
#    ini1 = list(result.x)
#    y = points.calc(state)
#    y = y.reshape((-1,6,2))
#    plt.figure()
#    for item in y:
#        plt.plot(*(item.T))
##    for item,value in zip(system.get_state_variables(),result.x):
##        initialvalues[item]=value
ini1=ini
    
#pAcm=pOA+lA/2*A.x
#pBcm=pAB+lB/2*B.x
pCcm=pOC+lC/2*C.x
pDcm=pCD+lD/2*D.x

#wOA = O.getw_(A)
#wAB = A.getw_(B)
wOC = O.getw_(C)
wCD = C.getw_(D)
#wBD = B.getw_(D)

BodyO = Body('BodyO',O,pOcm,mO,Dyadic.build(O,I_main,I_main,I_main),system)
#BodyA = Body('BodyA',A,pAcm,mA,Dyadic.build(A,I_leg,I_leg,I_leg),system)
#BodyB = Body('BodyB',B,pBcm,mB,Dyadic.build(B,I_leg,I_leg,I_leg),system)
#BodyC = Body('BodyC',C,pCcm,mC,Dyadic.build(C,I_leg,I_leg,I_leg),system)
#BodyD = Body('BodyD',D,pDcm,mD,Dyadic.build(D,I_leg,I_leg,I_leg),system)

#ParticleA = Particle(pAcm,mA,'ParticleA')
#ParticleB = Particle(pBcm,mB,'ParticleB')
ParticleC = Particle(pCcm,mC,'ParticleC')
ParticleD = Particle(pDcm,mD,'ParticleD')
#
#system.addforce(-b*wOA,wOA)
#system.addforce(-b*wAB,wAB)
system.addforce(-b*wOC,wOC)
system.addforce(-b*wCD,wCD)
#system.addforce(-b*wBD,wBD)
#
#stretch = -pBtip.dot(N.y)
#stretch_s = (stretch+abs(stretch))
#on = stretch_s/(2*stretch+1e-10)
#system.add_spring_force1(k_constraint,-stretch_s*N.y,vBtip)
#system.addforce(-b_constraint*vBtip*on,vBtip)

#system.add_spring_force1(k,(qA-qO)*N.z,wOA)
#system.add_spring_force1(k,(qB-qA)*N.z,wAB)
system.add_spring_force1(k,(qC-qO-preload3)*N.z,wOC)
system.add_spring_force1(k,(qD-qC-preload4)*N.z,wCD)
#system.add_spring_force1(k,(qD-qB-preload5)*N.z,wBD)


#system.addforcegravity(-g*N.y)

time_signal = sympy.tanh((system.t-1)*10)/2+.5 - sympy.tanh((system.t-3)*10)/2+.5- sympy.tanh((system.t-3)*10)/2+.5 + sympy.tanh((system.t-4)*10)/2+.5
torque = time_signal*stall_torque
#system.addforce(torque*O.z,wOA)
#system.addforce(-torque*O.z,wOC)

#
eq = []
#eq.append((pBtip-pDtip).dot(N.x))
#eq.append((pBtip-pDtip).dot(N.y))
#eq.append((O.y.dot(N.y)))
#eq.append(pOcm.dot(N.y)-initialvalues[y])
#eq.append(pOcm.dot(N.y)-initialvalues[y])
#eq.append(pOcm.dot(N.x)-initialvalues[x])
#eq.append(qO-initialvalues[qO])
#
eq_d= [system.derivative(item) for item in eq]
eq_dd= [system.derivative(item) for item in eq_d]
#
f,ma = system.getdynamics()
func1 = system.state_space_post_invert(f,ma,eq_dd,constants = system.constant_values)
states=pynamics.integration.integrate_odeint(func1,ini1,t,rtol=tol,atol=tol)

KE = system.get_KE()
PE = system.getPEGravity(0*N.x) - system.getPESprings()
energy = Output([KE-PE])
energy.calc(states)
energy.plot_time()

#torque_plot = Output([torque])
#torque_plot.calc(states)
#torque_plot.plot_time()

#points = [pDtip,pCD,pOC,pOA,pAB,pBtip]
points = [pDtip,pCD,pOC,pOA]
#points = [pOC,pOA,pAB,pBtip]
points = PointsOutput(points)
y = points.calc(states)
y = y.reshape((-1,6,2))
plt.figure()
for item in y[::30]:
    plt.plot(*(item.T))


#
#
#ini = states[-1]
#ini[2] = 0
#ini[7:] = 0
#ini = list(ini)
#
#func1 = system.state_space_post_invert(f,ma,constants = system.constant_values)
#states2=pynamics.integration.integrate_odeint(func1,ini,numpy.r_[tinitial:tfinal:1/30],hmax = .01,rtol=1e-3,atol=1e-3,args=({'constants':{},'alpha':1e3,'beta':1e1},))
#
#energy = Output([KE-PE])
#energy.calc(states2)
#energy.plot_time()
#
#tip = Output([pBtip.dot(N.y),stretch])
#tip.calc(states2)
#tip.plot_time()
#
points.animate(fps = 30, movie_name='render.mp4',lw=2)
