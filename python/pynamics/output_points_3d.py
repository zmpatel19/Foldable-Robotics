# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:30:38 2020

@author: danaukes
"""

class PointsOutput3D(PointsOutput):
    def __init__(self,y_exp, system=None, constant_values = None):
        system = system or pynamics.get_system()
        y_exp = [item for item2 in y_exp for item in [item2.dot(system.newtonian.x),item2.dot(system.newtonian.y),item2.dot(system.newtonian.z)]]
        Output.__init__(self,y_exp, system, constant_values)

    def calc(self,x):
        Output.calc(self,x)
        self.y.resize(self.y.shape[0],int(self.y.shape[1]/3),3)
        return self.y

    def animate(self,fps = 30,stepsize=1, movie_name = None,*args,**kwargs):
        # import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        from matplotlib import animation, rc

        y = self.y

        f = plt.figure()
        ax = f.add_subplot(1,1,1,aspect = 'equal',autoscale_on=False,projection='3d')
#        ax.axis('equal')
        limits   = [y[:,:,0].min(),y[:,:,0].max(),y[:,:,1].min(),y[:,:,1].max()]
        ax.axis(limits)


#        y = self.y[::stepsize]
        
        line, = ax.plot([], [], [],*args,**kwargs)
        
        def init():
#            line.set_data([], [],[])
            line.set_xdata([])
            line.set_ydata([])
            line.set_zdata([])
            return (line,)
        
        def run(item):
            xdata = item.T[0]
            ydata = item.T[1]
            zdata = item.T[2]
            line.set_xdata(xdata)
            line.set_ydata(ydata)
            line.set_zdata(zdata)
#            ax.axis('equal')
#            ax.axis(limits)
            return (line,)

        self.anim = animation.FuncAnimation(f, run, init_func=init,frames=y[::stepsize], interval=1/fps*1000, blit=True,repeat = True,repeat_delay=3000)        
        if movie_name is not None:
            self.anim.save(movie_name, fps=fps,writer='ffmpeg')