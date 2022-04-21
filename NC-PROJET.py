# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:59:37 2022

@author: cleme
"""

from numpy import *
#import numpy as np
from sympy import *

#import simpy as sp
import matplotlib.pyplot as plt

EL=-60
EK=-84
ECa=120
#--
V1=-1.2
V2=18
V3=2
V4=30
#--
gL=2
gK=8
gCa=4.4
#--
C=20
gamma=0.04
W=0.5
I=0
#def W():return 0.5

def m_inf(V):
    return (1+tanh((V-V1)/V2))/2

def w_inf(V):
    return (1+tanh((V-V2)/V4))/2    

def to_inf(V):
    return 1/(cosh((V-V3)/(2*V4)))

def ML_null1(V):
    return (-gCa*(0.5*(1+tanh((V-V1)/V2)))*(V-ECa)-gK*W*(V-ECa)-gL*(V-EL)+I)/C

def ML_null2(V):
    return gamma*(w_inf(V)-W/to_inf(V))

def V_nullcline(V):
    #return(-gCa*((1+tanh((V-V1)/V2))/2)*(V-ECa)-gL*(V-EL)+I)/gK*(V-EK)
    return (I - gL*(V-EL)-gCa*((1 + tanh((V -V1)/V2))/2)*(V-ECa))/(gK*(V-EK))


def V_nullcline1(V):
    return(-gCa*(0.5*(1+tanh((V-V1)/V2)))*(V-ECa)-gL*(V-EL)+0)/(gK*(V-EK))
def V_nullcline2(V):
    return(-gCa*(0.5*(1+tanh((V-V1)/V2)))*(V-ECa)-gL*(V-EL)+20)/(gK*(V-EK))
def V_nullcline3(V):
    return(-gCa*(0.5*(1+tanh((V-V1)/V2)))*(V-ECa)-gL*(V-EL)+60)/(gK*(V-EK))


V = linspace(-65,20,100000)


# color=['r-', 'b-', 'y-', 'g-', 'p-']
# for I in range(0,100,10):
#     print(I)
#     plt.plot(V, V_nullcline(V), color[I%20])
# plt.legend(['MN_null1', 'MN_null2'])
# plt.title('Le temps passe et la mort approche')
# plt.show()



#y2=w_inf(V)
# y3 = V_nullcline1(V)
# y4 = V_nullcline2(V)
# y5 = V_nullcline3(V)



# plt.plot(V, y2, 'b-')
# plt.plot(V, y3, 'y-')
# plt.plot(V, y4, 'black')
# plt.plot(V, y5, 'g-')

plt.xlabel('V')
plt.ylabel('W(V)')
plt.legend(['W(t)', 'V(t) : I=0', 'V : I=20', 'V : I=60'])
plt.title("Tracé des isoclines de W et de V pour différentes valeurs de I")
plt.show()


    
class ML():
    """modelisation of Moris Lecar model
    """
    def __init__(self):
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
    
    def isoclines(self, I):
        """draw model's isoclines
        """
        pass
    
    #==========================================================================
    def m_inf(self, V):
        return (1+tanh((V-self.V1)/self.V2))/2

    def w_inf(self, V):
        return (1+tanh((V-self.V2)/self.V4))/2    

    def to_inf(self, V):
        return 1/(cosh((V-self.V3)/(2*self.V4)))

    def ML_null1(self, V):
        return (-self.gCa*(0.5*(1+tanh((V-self.V1)/self.V2)))*(V-self.ECa)-self.gK*self.W*(V-self.ECa)-self.gL*(V-self.EL)+self.I)/self.C

    def ML_null2(self, V):
        return self.gamma*(self.w_inf(V)-self.W/self.to_inf(V))

    def V_nullcline(self, V):
        #return(-gCa*((1+tanh((V-V1)/V2))/2)*(V-ECa)-gL*(V-EL)+I)/gK*(V-EK)
        return (self.I - self.gL*(V-self.EL)-self.gCa*((1 + tanh((V -self.V1)/self.V2))/2)*(V-self.ECa))/(self.gK*(V-self.EK))
    #==========================================================================
    
    def V_intersept(self, iso1, iso2):
        """return nulcline intersection coordinates.
        """
        #https://askcodez.com/intersection-de-deux-graphes-en-python-trouvez-la-valeur-x.html
        Vl = linspace(-65, -40, 1000)
        Vr = linspace(-40, 20, 1000)
        V = linspace(-65, 20, 1000)
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
        return 
    

    def jacobienne(self, I):
        v = Symbol("v")
        w = Symbol("w")
        dw = self.gamma * (((1 + tanh((v - self.V2)/self.V4))/2)- w)/((cosh((v - self.V3)/(2*self.V4)))**-1)
        dv = -self.gCa * ((1 + tanh((v - self.V1)/self.V2))/2) * (v - self.ECa) - self.gK * w * (v - self.ECa) - self.gL* (v-self.EL) + self.I
        J = [[dv.diff(v), "-------------", dv.diff(w)],[dw.diff(v), "-------------", dw.diff(w)]]
        return J
        
    def J(self, v, w):
        return array([[-8*w + (v - 120)*(0.122222222222222*tanh(v/18 + 0.0666666666666667)**2 - 0.122222222222222) - 2.2*tanh(v/18 + 0.0666666666666667) - 4.2,
          960 - 8*v], 
         [(0.000666666666666667 - 0.000666666666666667*tanh(v/30 - 3/5)**2)*cosh(v/60 - 1/30) + (-0.04*w + 0.02*tanh(v/30 - 3/5) + 0.02)*sinh(v/60 - 1/30)/60, 
          -0.04*cosh(v/60 - 1/30)]],dtype=float)



if __name__ == '__main__':
    
    V = linspace(-65,20,1000) #abscisse
    ML=ML() #modèle Moris Lecart
    #idx = ML.V_intersept(w_inf, V_nullcline) #points d'équilibre
    
    #--------------Equilibre 0-----------###
    j = ML.J(-1, 0.18)
    det = linalg.det(j)
    vp = linalg.eigvals(j)
    print(vp[0]*vp[1])
    #print(ML.dV())
    
    #Equilibre 1###
    j2 = ML.J(3, 0.27)
    det2 = linalg.det(j2)
    vp = linalg.eigvals(j2)
    print(vp[0]*vp[1])
    #print(ML.dV())
    
    #Equilibre 2###
    j3 = ML.J(-60,0.01)
    det3 = linalg.det(j3)
    vp = linalg.eigvals(j3)
    print(vp[0])
    
    l=[det,det2,det3]
    for i in range (len(l)):
        if l[i] > 0:
            print(f"j {i} n'est pas un point selle")
        else:
            print(f"j {i} est un point selle")
    #print(ML.dV())
   
          
    
    

    
    













