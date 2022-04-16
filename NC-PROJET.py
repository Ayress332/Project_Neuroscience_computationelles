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
    return 0.5*(1+tanh((V-V1)/V2))

def w_inf(V):
    return 0.5*(1+tanh((V-V2)/V4))

def to_inf(V):
    return 1/(cosh((V-V3)/(2*V4)))

def ML_null1(V):
    return (-gCa*(0.5*(1+tanh((V-V1)/V2)))*(V-ECa)-gK*W*(V-ECa)-gL*(V-EL)+I)/C

def ML_null2(V):
    return gamma*(w_inf(V)-W/to_inf(V))


V = linspace(-100,100,10000)
y1=ML_null1(V)
y2=ML_null2(V)


plt.plot(V, y1, 'r-')
plt.plot(V, y2, 'b-')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['MN_null1', 'MN_null2'])
plt.title('Le temps passe et la mort approche')
plt.show()
