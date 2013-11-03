from mumax2 import *
from math import *
from mumax2_geom import *

eps = 1e-6
# Test for LLB with 2TM
# Ni, just like in PRB 81, 174401 (2010)
  
Nx = 64
Ny = 64
Nz = 64

sX = 320e-9
sY = 160e-9
sZ = 80e-9

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)

loadargs('mfa/msat0', ["T:Te"], [], [])
loadargs('mfa/Ts',["msat:msat0"],[],[])

setv('Tc', 633.0)
setv('J', 0.308)
setv('n', 9.14e28)

Msat = 548e3

setv('msat0T0', Msat)
setv('msat0', Msat)
setv('msat', Msat)

Mmask = [[[[1.0]]]]
setmask('msat', Mmask)
setmask('msat0', Mmask)
setmask('msat0T0', Mmask)

m = [[[[1.0]]],[[[0.0]]],[[[0.0]]]]
setarray('mf', m)

Te = 100.0
T = [[[[Te]]]]
setarray('Te', T)

Ms = getcell('msat0', 32, 32, 32)[0]
print Ms

Ts = getcell('Ts', 32, 32, 32)[0]
print Te, Ts, abs(Ts-Te)/(Ts+Te)

