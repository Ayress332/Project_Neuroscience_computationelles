# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 13:12:43 2022

@author: cleme
"""

from numpy import *
from sympy import *
import matplotlib.pyplot as plt


class ML():
    """modelisation of Moris Lecar model
    """
    def __init__(self):
        self.V = linspace(-65,20,1000)
        #----initialisation of all parameters
        self.EL=-60
        self.EK=-84
        self.ECa=120
        #--
        self.V1=-1.2
        self.V2=18
        self.V3=2
        self.V4=30
        #--
        self.gL=2
        self.gK=8
        self.gCa=4.4
        #--
        self.C=20
        self.gamma=0.04
        self.W=0.5
        self.I=0
    
    #==========================================================================
    # Definition of all model's functions 
    def m_inf(self, V):
        return (1+tanh((V-self.V1)/self.V2))/2

    def w_inf(self, V):
        return (1+tanh((V-self.V2)/self.V4))/2 

    def to_inf(self, V):
        return 1/(cosh((V-self.V3)/(2*self.V4)))

    def V_null(self, V):
        return (-self.gCa*(0.5*(1+tanh((V-self.V1)/self.V2)))*(V-self.ECa)-self.gK*self.W*(V-self.ECa)-self.gL*(V-self.EL)+self.I)/self.C

    def W_null(self, V):
        return self.gamma*(self.w_inf(V)-self.W/self.to_inf(V))

    def V_nullcline(self, V):
        #return(-gCa*((1+tanh((V-V1)/V2))/2)*(V-ECa)-gL*(V-EL)+I)/gK*(V-EK)
        return (self.I - self.gL*(V-self.EL)-self.gCa*((1 + tanh((V -self.V1)/self.V2))/2)*(V-self.ECa))/(self.gK*(V-self.EK))
    
    #==========================================================================
    def isoclines(self, Imin, Imax, nI):
        """draw model's isoclines, for n different values of I between Imin and Imax
        """
        V=self.V
        color=['r-', 'b-', 'y-', 'g-', 'p-']
        for I in range(Imin, Imax+1, (Imax+1-Imin)//nI ):
            plt.plot(V, self.V_null(V), color[I%len(color)])
        plt.legend(['W(t)', 'V(t) : I=0', 'V : I=20', 'V : I=60'])
        plt.title("Tracé des isoclines de W et de V pour différentes valeurs de I")
        plt.show()
        pass
    
    def V_intersept(self, iso1, iso2):
        """return nulcline intersection coordinates.
        """
        #https://askcodez.com/intersection-de-deux-graphes-en-python-trouvez-la-valeur-x.html
        Vl = linspace(-65, -40, 1000)
        Vr = linspace(-40, 20, 1000)
        V = linspace(-65, 20, 100)
        f = iso2(V)
        g = iso1(V)
        
        plt.plot(V, f, '-')
        plt.plot(V, g, '-')
        
        idx = argwhere(diff(sign(f - g)) != 0).reshape(-1) + 0
        print(f"""
              a0 : ({idx[0]}, {iso1(idx[0])}, {iso2(idx[0])})
              a1 : ({idx[1]}, {iso1(idx[1])})
              a2 : ({idx[2]}, {iso1(idx[2])})
              """)
        plt.plot(V[idx], f[idx], 'ro')
        plt.show()
        intersections_l = [(Vl[i], f[i]) for i,_ in enumerate(zip(f,g)) if abs(f[i]-g[i])<10**-4]
        intersections_r = [(Vr[i], f[i]) for i,_ in enumerate(zip(f,g)) if abs(f[i]-g[i])<10**-4]
        print(intersections_l, intersections_r)
        print(idx)
        return intersections_l, intersections_r
    

    def jacobienne(self, I):
        """Returne the symbolic expression of the ML system's Jacobienne
        """
        v = Symbol("v")
        w = Symbol("w")
        dw = self.gamma * (((1 + tanh((v - self.V2)/self.V4))/2)- w)/((cosh((v - self.V3)/(2*self.V4)))**-1)
        dv = -self.gCa * ((1 + tanh((v - self.V1)/self.V2))/2) * (v - self.ECa) - self.gK * w * (v - self.ECa) - self.gL* (v-self.EL) + self.I
        J = [[dv.diff(v), "-------------", dv.diff(w)],[dw.diff(v), "-------------", dw.diff(w)]]
        return J
    
    def ML_pros(self):
        """
        """
        v = Symbol("v")
        w = Symbol("w")
        dw = self.gamma * (((1 + tanh((v - self.V2)/self.V4))/2)- w)/((cosh((v - self.V3)/(2*self.V4)))**-1)
        dv = -self.gCa * ((1 + tanh((v - self.V1)/self.V2))/2) * (v - self.ECa) - self.gK * w * (v - self.ECa) - self.gL* (v-self.EL) + self.I
        return [Integral(dw, v)]
        
    def J(self, v, w):
        """Calculate the Jacobienne of ML system, based on ML.jacobienne() output expression
        """
        return array([[-8*w + (v - 120)*(0.122222222222222*tanh(v/18 + 0.0666666666666667)**2 - 0.122222222222222) - 2.2*tanh(v/18 + 0.0666666666666667) - 4.2,
          960 - 8*v], 
         [(0.000666666666666667 - 0.000666666666666667*tanh(v/30 - 3/5)**2)*cosh(v/60 - 1/30) + (-0.04*w + 0.02*tanh(v/30 - 3/5) + 0.02)*sinh(v/60 - 1/30)/60, 
          -0.04*cosh(v/60 - 1/30)]],dtype=float)

    def steady_point(self):
        """Determinates the nature of steady states
        """
        equilibre = V_intersept(self.w_inf(V), self.V_nullcline())
        det = [] #déterminants
        vp = [] #valeur propres
        tr = [] #trace
        #----remplissage de boucle
        
        #----interprétation des résultats
        for i in range (len(l)):
            if l[i] > 0:
                print(f"j {i} n'est pas un point selle")
            else:
                print(f"j {i} est un point selle")



if __name__ == '__main__':
    
    ml = ML()
    ml.isoclines(0,100,10)
   
