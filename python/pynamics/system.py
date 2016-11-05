# -*- coding: utf-8 -*-
"""
Written by Daniel M. Aukes
Email: danaukes<at>gmail.com
Please see LICENSE for full license.
"""

import sympy
import pynamics
import numpy
import scipy
import pydevtools.svd as svd

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
    
class System(object):
    _z = 0
    def __init__(self):
        self.derivatives = {}
        self.constants = {}
        self.forces = []
        self.effectiveforces = []
        self.momentum = []
        self.KE = sympy.Number(0)
        self.bodies = []
        self.particles = []
        self.q = {}
        self.replacements = {}
        self.springs = []

    def add_q(self,q,ii):
        if ii in self.q:
            self.q[ii].append(q)
        else:
            self.q[ii] = [q]

    def get_q(self,ii):
        if ii in self.q:
            return self.q[ii]
        else:
            return []
            
    def set_newtonian(self,frame):
        self.newtonian = frame
        
    def generatez(self,number):
        z=sympy.Symbol('z'+str(self._z))
        self.replacements[z]=number
        self._z+=1
        return z

    def addforce(self,force,velocity):
        self.forces.append((force,velocity))

    def add_spring_force(self,k,stretch,velocity):
        force = -k*stretch
        self.forces.append((force,velocity))
        self.springs.append((k,stretch))

    def addeffectiveforce(self,effectiveforce,velocity):
        self.effectiveforces.append((effectiveforce,velocity))

    def addmomentum(self,momentum,velocity):
        self.momentum.append((momentum,velocity))

    def addKE(self,KE):
        self.KE+=KE

    def getPEGravity(self,point):
        PE = pynamics.ZERO
        for body in self.bodies+self.particles:
            if body.gravityvector is not None:
                d = body.pCM - point
                F = body.forcegravity
                PE +=F.dot(d)
        return PE

    def getPESprings(self):
        PE = pynamics.ZERO
        for k,stretch in self.springs:
            PE+=.5*k*stretch.dot(stretch)
        return PE
        
    def addforcegravity(self,gravityvector):
        for body in self.bodies:
            body.addforcegravity(gravityvector)
        for particle in self.particles:
            particle.addforcegravity(gravityvector)

    def getdynamics(self):
        q_d = self.get_q(1)
        generalizedforce=self.generalize(self.forces,q_d)
        generalizedeffectiveforce=self.generalize(self.effectiveforces,q_d)
#        return 
#        zero = {}
#        for speed in self.get_q(1):
#            zero[speed] = generalizedeffectiveforce[speed] - generalizedforce[speed]
        return generalizedforce,generalizedeffectiveforce

    def solvedynamics(self,f,ma,method = 'LU',auto_z= False):
        f = sympy.Matrix(f)
        ma = sympy.Matrix(ma)
        Ax_b = ma-f
        x = sympy.Matrix(self.get_q(2))
        A = Ax_b.jacobian(x)
#        replacement = dict([(str(item.func),sympy.Symbol(str(item.func))) for item in self.get_q(0)+self.get_q(1)])
#        fAx_b = sympy.lambdify(self.get_q(2),Ax_b,modules = ['math','mpmath','numpy',replacement])
#        b = -fAx_b(*(0*x))
#        b = (A*x-Ax_b).expand()
        b = -Ax_b.subs(dict([(item,0) for item in self.get_q(2)]))
        if auto_z:
            def func1(x):
                if x!=pynamics.ZERO:
                    return self.generatez(x)
                else:
                    return x
            AA = A.applyfunc(func1)
            AA_inv = AA.inv(method = method)
            keys = self.replacements.keys()
            fAA_inv = sympy.lambdify(keys,AA_inv)
            A_inv = fAA_inv(*[self.replacements[key] for key in keys])
#            A_inv = AA_inv.subs(self.replacements)
        else:
            A_inv = A.inv(method=method)
        return A_inv*b 
        
    def generalize(self,list1,q_d):
        generalized=[]
        for speed in q_d:
            new = pynamics.ZERO
#            generalized[speed]=pynamics.ZERO
            for expression,velocity in list1:
                new+=expression.dot(velocity.diff_partial_local(speed,self))
            generalized.append(new)
        return generalized
        
    def add_derivative(self,expression,variable):
        self.derivatives[expression]=variable
    def add_constant(self,constant,value):
        self.constants[constant]=value

    def createsecondorderfunction(system,f,ma):
        statevariables = system.get_q(0)+system.get_q(1)
        var_dd = system.solvedynamics(f,ma,auto_z=True)
        var_dd=var_dd.subs(system.constants)
        functions = [sympy.lambdify(statevariables,rhs) for rhs in var_dd]
        indeces = [statevariables.index(element) for element in system.get_q(1)]
        def func1(state,time):
            return numpy.r_[[state[ii] for ii in indeces],[f(*state) for f in functions]].tolist()
        return func1

    def createsecondorderfunction2(system,f,ma):
        q_state = system.get_q(0)+system.get_q(1)
        q_state_d = system.get_q(1) + system.get_q(2)
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
        A_full = sympy.zeros(2*m)   
        A_full[:m,:m] = sympy.eye(m)
        A_full[m:,m:] = A
        b_full = sympy.zeros(2*m,1)
        b_full[:m,0]=q_d
        b_full[m:,0]=b
        
        c_sym = list(system.constants.keys())
        c_val = [system.constants[key] for key in c_sym]
    
        fA = sympy.lambdify(q_state+c_sym,A_full)
        fb = sympy.lambdify(q_state+c_sym,b_full)
    
        @static_vars(ii=0)
        def func(state,time):
            a = list(state)+c_val
            Ai = fA(*a)
            bi = fb(*a)
            if func.ii%1000==0:
                print(time)
            func.ii+=1
            x = numpy.array(scipy.linalg.inv(Ai).dot(bi))
            return x.flatten().tolist()
        return func

    def createsecondorderfunction3(system,f,ma):
        q_state = system.get_q(0)+system.get_q(1)
        q_state_d = system.get_q(1) + system.get_q(2)
    #    q = system.get_q(0)+system.get_q(1)
        f = sympy.Matrix(f)
#        f = f.subs(system.constants)
    #    f_f = sympy.lambdify(q,f)
        ma = sympy.Matrix(ma)
    #    ma = ma.subs(system.constants)
    #    f_ma = sympy.lambdify(q,f)
        q = system.get_q(0)
        q_d = system.get_q(1)
        q_dd = system.get_q(2)
    
        Ax_b = ma-f
        Ax_b = Ax_b.subs(system.constants)
        x = sympy.Matrix(q_dd)
        A = Ax_b.jacobian(x)
        b = -Ax_b.subs(dict(list([(item,0) for item in x])))
        
        m = len(q_d)
        A_full = sympy.zeros(2*m)   
        A_full[:m,:m] = sympy.eye(m)
        A_full[m:,m:] = A
        b_full = sympy.zeros(2*m,1)
        b_full[:m,0]=q_d
        b_full[m:,0]=b
        
#        c_sym = list(system.constants.keys())
#        c_val = [system.constants[key] for key in c_sym]
    
        fA = sympy.lambdify(q_state,A_full)
        fb = sympy.lambdify(q_state,b_full)
    
    
        def func(state,time):
            a = state
            x = numpy.array(scipy.linalg.inv(fA(*a))*fb(*a))
            return x.flatten().tolist()
        return func
        
    def createsecondorderfunction4(system,f,ma,fJ):
        q_state = system.get_q(0)+system.get_q(1)
        q_state_d = system.get_q(1) + system.get_q(2)
    #    q = system.get_q(0)+system.get_q(1)
        f = sympy.Matrix(f)
    #    f = f.subs(system.constants)
    #    f_f = sympy.lambdify(q,f)
        ma = sympy.Matrix(ma)
    #    ma = ma.subs(system.constants)
    #    f_ma = sympy.lambdify(q,f)
        q = system.get_q(0)
        q_d = system.get_q(1)
        q_dd = system.get_q(2)
    
        Ax_b = ma-f
    #    Ax_b = Ax_b.subs(system.constants)
        x = sympy.Matrix(q_dd)
        A = Ax_b.jacobian(x)
        b = -Ax_b.subs(dict(list([(item,0) for item in x])))
        
        m = len(q_d)
        A_full = sympy.zeros(2*m)   
        A_full[:m,:m] = sympy.eye(m)
        A_full[m:,m:] = A
        b_full = sympy.zeros(2*m,1)
        b_full[:m,0]=q_d
        b_full[m:,0]=b
        
        c_sym = list(system.constants.keys())
        c_val = [system.constants[key] for key in c_sym]
    
        fA = sympy.lambdify(q_state+c_sym,A_full)
        fb = sympy.lambdify(q_state+c_sym,b_full)
    
        @static_vars(ii=0)
        def func(state,time):
            a = list(state)+c_val
#            global ii
            Ji = fJ(*state)
            U, s, V = scipy.linalg.svd(Ji,full_matrices = False)

            s= s[s!=0]
            V = V[s!=0]
            l = len(s)

            Ai = fA(*a)
            bi = fb(*a)
            if func.ii%100==0:
                print(time)
            func.ii+=1
#            A = numpy.vstack((Ai[4:,4:],Ji))            
            
#            bi[m:] = bi[m:] - (s*V).T
            n = len(Ai)
            Ai2 = numpy.zeros((n+l,n+l))
            Ai2[:n,:n] = Ai
            Ai2[n:,m:n] = V
            Ai2[m:n,n:] = V.T
#            Ai2[n:,n:] = numpy.eye(l)
            bi2 = numpy.zeros((n+l,1))
            bi2[:n] = bi[:n]
#            bi2[n:] = numpy.zeros(l)
#            U, s, V = scipy.linalg.svd(A,full_matrices = True)
#            print(U.shape,s.shape,V.shape)
#            Ai[4:,4:]=V.T
            x = numpy.array(scipy.linalg.inv(Ai2).dot(bi2))
            
#            x =             
            return x.flatten().tolist()
        return func
        
    def createsecondorderfunction5(system,f,ma,fJ,zero):
        q_state = system.get_q(0)+system.get_q(1)
        q_state_d = system.get_q(1) + system.get_q(2)
    #    q = system.get_q(0)+system.get_q(1)
        f = sympy.Matrix(f)
        f = f.subs(system.constants)
    #    f_f = sympy.lambdify(q,f)
        ma = sympy.Matrix(ma)
        ma = ma.subs(system.constants)
    #    f_ma = sympy.lambdify(q,f)
        q = system.get_q(0)
        q_d = system.get_q(1)
        q_dd = system.get_q(2)
    
        Ax_b = ma-f
    #    Ax_b = Ax_b.subs(system.constants)
        x = sympy.Matrix(q_dd)
        A = Ax_b.jacobian(x)
        b = -Ax_b.subs(dict(list([(item,0) for item in x])))
        fA = sympy.lambdify(q_state,A)
        fb = sympy.lambdify(q_state,b)

        z = zero.subs(system.constants)
        fz = sympy.lambdify(q_state,z)
        
        
        @static_vars(ii=0)
        def func(state,time):
            Ai = fA(*state)
            bi = fb(*state)
            Ji = fJ(*state)
            zi = fz(*state)

#            U, s, V = scipy.linalg.svd(Ji,full_matrices = False)
#            if 0 in s:
#                print(s)
#                s = s[s!=0]
#                U = U[:,s!=0]
#                V = V[s!=0]
#                S = numpy.zeros((U.shape[1],V.shape[0]))
#                S[:len(s),:len(s)] = numpy.diag(s)
#                Ji = numpy.matrix(S.dot(V))
            
            if func.ii%1000==0:
                print(time)
            func.ii+=1
            
            a_state = numpy.c_[state]

            m_A,n_A = Ai.shape
            m_b,n_b = bi.shape
            m_J,n_J = Ji.shape

            A_1 = numpy.vstack((Ai,Ji))
            C = numpy.zeros((m_J,m_J))
            A_2 = numpy.vstack((Ji.T,C))
            A = numpy.hstack((A_1,A_2))

#            U, s, V = scipy.linalg.svd(A,full_matrices = False)
#            filt = abs(s)/abs(s).max()>1e-3
##            S = numpy.zeros((U.shape[1],V.shape[0]))
##            S[:len(s),:len(s)] = numpy.diag(s)
#            S_pinv = numpy.zeros((U.shape[1],V.shape[0]))
#            S_pinv[filt,filt] = 1/s[filt]
#            S_pinv = S_pinv.T
#            
#            A_inv = ((V.T.dot(S_pinv)).dot(U.T))
            A_inv = scipy.linalg.pinv2(A_1)
            m_q = len(state)
            A_inv_full = numpy.zeros((m_q+m_J,m_q+m_J))
            A_inv_full[:m_q-m_b,:m_q-m_b] = numpy.eye(m_q-m_b)
            A_inv_full[m_q-m_b:,m_q-m_b:] = A_inv
            b_full = numpy.zeros((m_q+m_J,1))
            b_full[:m_q-m_b] = a_state[m_q-m_b:]
            b_full[m_q-m_b:m_q] = bi
            x = A_inv_full.dot(b_full)
#            x[m_q-m_b:m_q] += -1e-1*Ji.T*zi
#            x[:m_q-m_b] += -1e-2*Ji.T*zi
#            x
            
#            x = A_full
#            A = numpy.vstack((Ai[4:,4:],Ji))            
#            bi[m:] = bi[m:] - (s*V).T
#            n = len(Ai)
#            Ai2 = numpy.zeros((n+l,n+l))
#            Ai2[:n,:n] = Ai
#            Ai2[n:,m:n] = V
#            Ai2[m:n,n:] = V.T
##            Ai2[n:,n:] = numpy.eye(l)
#            bi2 = numpy.zeros((n+l,1))
#            bi2[:n] = bi[:n]
##            bi2[n:] = numpy.zeros(l)
##            U, s, V = scipy.linalg.svd(A,full_matrices = True)
##            print(U.shape,s.shape,V.shape)
##            Ai[4:,4:]=V.T
#            x = numpy.array(scipy.linalg.inv(Ai2).dot(bi2))
            
#            x =             
            return x[:m_q].flatten().tolist()
#            return [0]*len(state)
        return func
    def createsecondorderfunction6(system,f,ma,fJ,zero,zero_d):
        from numpy import r_, c_
        q_state = system.get_q(0)+system.get_q(1)
    #    q = system.get_q(0)+system.get_q(1)
        f = sympy.Matrix(f)
        f = f.subs(system.constants)
    #    f_f = sympy.lambdify(q,f)
        ma = sympy.Matrix(ma)
        ma = ma.subs(system.constants)
    #    f_ma = sympy.lambdify(q,f)
        q_dd = system.get_q(2)
    
        Ma_f = ma-f
        M = Ma_f.jacobian(q_dd)
        f = -Ma_f.subs(dict(list([(item,0) for item in q_dd])))
        fM = sympy.lambdify(q_state,M)
        ff = sympy.lambdify(q_state,f)

        z = zero.subs(system.constants)
        fz = sympy.lambdify(q_state,z)

        z_d = zero_d.subs(system.constants)
        fz_d = sympy.lambdify(q_state,z_d)
        
        
        @static_vars(ii=0)
        def func(state,time):
            Mi = fM(*state)
            fi = ff(*state)
            Ji = fJ(*state)
            zi = fz(*state)
            z_di = fz_d(*state)

            Mi_inv = scipy.linalg.pinv(Mi)

            if func.ii%1000==0:
                print(time)
            func.ii+=1
            
#            Ji_o = Ji            
#            Ji = svd.prefilter(Ji)
            
            k,l = M.shape
            m,n = Ji.shape
            A = Ji.dot(Mi_inv.dot(Ji.T))
            b = -Ji.dot(Mi_inv.dot(fi))
#            Ai = svd.pinv(A)
            Ai = scipy.linalg.pinv2(A)
#            if not ((Ai-Ai2)==0).all():
#                print(Ai-Ai2)
#                Ai = svd.pinv(A)
#            else:
#                Ai = svd.pinv(A)
            lam = Ai.dot(b)
            
#            H = r_[c_[Mi,-Ji.T],c_[-Ji,numpy.zeros((m,m))]]
#            zb = r_[numpy.zeros(k),numpy.asarray(b).squeeze()]
#            ylam = scipy.linalg.pinv2(H).dot(zb)
#            lam = ylam[k:]
            a = numpy.asarray(Mi_inv.dot(Ji.T.dot(lam)+fi)).flatten()
            a += numpy.asarray(-1e4*Ji.T*z_di).flatten()
#            print(a,b)            
            a = a.tolist()

            v = numpy.asarray(state[k:]).flatten()
            v += numpy.asarray(-1e4*Ji.T*zi).flatten()
            v = v.tolist()
            return v+a
            
        return func


    def create_state_space_constrained(system,f,ma,eq = None,eq_active = None):
        eq = eq or []
        
        q = system.get_q(0)
        q_d = system.get_q(1)
        q_dd = system.get_q(2)
    
        if not eq:
            J = sympy.Matrix([])
            c = sympy.Matrix
        else:
            eq2 = sympy.Matrix([eq])
            J = eq2.jacobian(q_dd)
            c = (eq2-J*sympy.Matrix(q_dd)).expand()
            
        q_state = system.get_q(0)+system.get_q(1)
        f = sympy.Matrix(f)
        ma = sympy.Matrix(ma)
    
        Ax_b = ma-f
        x = sympy.Matrix(q_dd)
        A = Ax_b.jacobian(x)
        b = -Ax_b.subs(dict(list([(item,0) for item in x])))
        
        m = len(q_d)
    
        if not eq:
            A_full = A
            b_full = b
        
        else:
            n = len(eq)
        
            A_full = sympy.zeros(m+n)   
            A_full[:m,:m] = A
            A_full[m:,:m] = J
            A_full[:m,m:] = J.T
        
            b_full = sympy.zeros(m+n,1)
            b_full[:m,0]=b
            b_full[m:,0]=-c
            
        c_sym = list(system.constants.keys())
        c_val = [system.constants[key] for key in c_sym]
    
        fA = sympy.lambdify(q_state+c_sym,A_full)
        fb = sympy.lambdify(q_state+c_sym,b_full)
    
        def func(state,time,*args):
    #        print(args)
            a = list(state)+c_val
            Ai = fA(*a)
            bi = fb(*a)
            x1 = state[m:]
            x2 = numpy.array(scipy.linalg.inv(Ai).dot(bi)).flatten()
            x3 = numpy.r_[x1,x2[:m]]
            x4 = x3.flatten().tolist()
            return x4
        return func
    
    @staticmethod
    def assembleconstrained(eq_dyn,eq_con,q_dyn,q_con,method='LU'):
        AC1x_b1 = sympy.Matrix(eq_dyn)
        C2x_b2 = sympy.Matrix(eq_con)
        print('Ax-b')
        
        q_dyn = sympy.Matrix(q_dyn)
        q_con = sympy.Matrix(q_con)
        x = q_dyn.col_join(q_con)
        print('x,l')
        
        MASS = AC1x_b1.jacobian(q_dyn)
        C1 = AC1x_b1.jacobian(q_con)
        C2 = C2x_b2.jacobian(x)
        AA = sympy.Matrix.col_join(sympy.Matrix.row_join(MASS,C1),C2)
        print('A,C1,C2')
        
        b1 = -AC1x_b1.subs(zip(x.T.tolist()[0],[0 for item in x]))
        b2 = -C2x_b2.subs(zip(x.T.tolist()[0],[0 for item in x]))
        b = b1.col_join(b2)
        return AA,b,x    
    @classmethod
    def solveconstraineddynamics(cls,eq_dyn,eq_con,q_dyn,q_con,method='LU'):
        AA,b,x = cls.assembleconstrained(eq_dyn,eq_con,q_dyn,q_con,method=method)
    #    AC1x_b1 = sympy.Matrix(eq_dyn)
    #    C2x_b2 = sympy.Matrix(eq_con)
    #    print 'Ax-b'
    #    
    #    q_dyn = sympy.Matrix(q_dyn)
    #    q_con = sympy.Matrix(q_con)
    #    x = q_dyn.col_join(q_con)
    #    print 'x,l'
    #    
    #    MASS = AC1x_b1.jacobian(q_dyn)
    #    C1 = AC1x_b1.jacobian(q_con)
    #    C2 = C2x_b2.jacobian(x)
    #    AA = sympy.Matrix.col_join(sympy.Matrix.row_join(MASS,C1),C2)
    #    print 'A,C1,C2'
    #    
    #    b1 = -AC1x_b1.subs(zip(x.T.tolist()[0],[0 for item in x]))
    #    b2 = -C2x_b2.subs(zip(x.T.tolist()[0],[0 for item in x]))
    #    b = b1.col_join(b2)
        
        AA_inv = AA.inv(method = method)
        xx = AA_inv*b
        x_dyn = xx[0:len(q_dyn),:]
        x_con = xx[len(q_dyn):,:]
        return x_dyn,x_con    

    def derivative(self,expression):
        for ii,a in enumerate(self.derivatives.keys()):
            if ii==0:
                result = expression.diff(a)*self.derivatives[a]
            else:
                result += expression.diff(a)*self.derivatives[a]
        return result
    def state_variables(self):
        return self.get_q(0)+self.get_q(1)
    def state_variables_d(self):
        return self.get_q(1)+self.get_q(2)

                        