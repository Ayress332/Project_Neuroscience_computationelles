# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:59:37 2022

@author: cleme
"""

from numpy import *

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
#     plt.plot(V, V_nullclineI(V, I), color[I%20])
# plt.legend(['MN_null1', 'MN_null2'])
# plt.title('Le temps passe et la mort approche')
# plt.show()


y1=V_nullcline(V)
y2=w_inf(V)
y3 = V_nullcline1(V)
y4 = V_nullcline2(V)
y5 = V_nullcline3(V)


plt.plot(V, y1, 'r-')
plt.plot(V, y2, 'b-')
plt.plot(V, y3, 'y-')
plt.plot(V, y4, 'black')
plt.plot(V, y5, 'g-')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(['MN_null1', 'MN_null2'])
plt.title('Le temps passe et la mort approche')
plt.show()


    


























