El = -60
Ek = -84
Eca = 120
V1 = -1.2
V2 = 18
V3 = 2
V4 = 30
gl = 2
gk = 8
gCa = 4.4
C = 20
gama = 0.04

function [m]=minf(V)
    m = (1+tanh((V-V1)/V2))/2
endfunction

function [W]=winf(V)
    W = (1+tanh((V-V2)/V4))/2
endfunction

function [t]=tinf(V)
    t = 1/(cosh((V-V3)/(2*V4)))
endfunction
