import numpy as np
import matplotlib.pyplot as plt
from sympy import *

EL = -60 ; EK = -84 ; ECa = 120
V1 = -1.2 ; V2 = 18 ; V3 = 2 ; V4 = 30
gL = 2 ; gK = 8 ; gCa = 4.4
C = 20 ; gamma = 10**(-20)

def derive_v(choice, I): #dérivée de la première fonction
  v = Symbol("v")
  w = Symbol("w")
  f = -gCa * ((1 + tanh((v - V1)/V2))/2) * (v - ECa) - gK * w * (v - ECa) - gL* (v-EL) + I
  if choice == 'v':
    return f.diff(v)
  if choice == 'w':
    return f.diff(w)

def derive_w(choice): #dérivée de la seconde fonction
  v = Symbol("v")
  w = Symbol("w")
  f = gamma * (((1 + tanh((v - V2)/V4))/2)- w)/((cosh((v - V3)/(2*V4)))**-1)
  if choice == 'v':
    return f.diff(v)
  if choice == 'w':
    return f.diff(w)

def V(I,v): #V nullcline
  w=(I - gL*(v-EL)-gCa*((1 + np.tanh((v -V1)/V2))/2)*(v-ECa))/(gK*(v-EK))
  return w

def W(I,v): #W nullcline
  w=(1+np.tanh((v-V2)/V4))/2
  return w

def plot(I): #plot V nullcline et W nullcline
  #intervalle
  v = np.linspace(-100,1,100)
  
  #functions
  y = V(I,v)
  z = W(I,v)

  # plot the function
  plt.plot(v,y, 'r')
  plt.plot(v,z, 'b')

  # show the plot
  plt.show()

if __name__ == '__main__':
  
  I = 5
  v = -60
  w = 0
  x1 = derive_v('v',I)
  x2 = derive_v('w',I)
  x3 = derive_w('v')
  x4 = derive_w('w')
  print(x1)
  print(x2)
  print(x3)
  print(x4)
  #print(a)



