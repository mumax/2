from mumax2 import *
from math import *
from mumax2_geom import *

# Test for LLB with 2TM
# Ni, just like in PRB 81, 174401 (2010)
  
Nx = 32
Ny = 32
Nz = 32

sX = 160e-9
sY = 160e-9
sZ = 160e-9

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)

# LLBR
load('exchange6')
load('demag')
load('zeeman')

load('llbr')
load('llbr/torque')
load('llbr/transverse')
load('temperature/LTM')

loadargs('dissipative-function',[], ["m:mf","R:llbr_transverse"],["Qmag:Qc"])
load('temperature/brown-anisotropic')

add_to('llbr_RHS', 'llbr_torque')
add_to('llbr_RHS', 'llbr_transverse')

add_to("Ql", "Qc")

load('solver/rk12')

setv('mf_maxerror', 1e-4)

setv('Temp_maxerror', 1e-4)

savegraph("graph.png")

Ms0 = 480e3

# Py

msat = [[[[1.0]]]]
setmask('msat', msat) 
setmask('msat0', msat)
setmask('msat0T0', msat)

mf =[ [[[1.0]]], [[[0.0]]], [[[0.0]]] ]
setarray('Mf', mf)

setv('msat', Ms0)        
setv('msat0', Ms0) 
setv('msat0T0', Ms0) 

Aex = 0.86e-11
setv('Aex', Aex)
setv('gamma_LL', 2.211e5)
setv('cutoff_dt', 1e-14)
Bx = 1.0 # 1 T
By = 0.0 
Bz = 0.0
setv('B_ext',[Bx,By,Bz])
                
# Heat bath parameters

setv('Cp_l', 3.0e6)

# Baryakhtar relaxation parameters

mu = 0.005
setv('mu', [mu, mu, mu])

tt = 1e-14
 
autotabulate(["t", "<Temp>"], "Temp.txt", tt)
autotabulate(["t", "<mf>"], "mf.txt", tt)
autotabulate(["t", "<Qc>"], "Qc.txt", tt)
autotabulate(["t", "<Ql>"], "Ql.txt", tt)
autotabulate(["t", "<H_therm>"], "H_therm.txt", tt)

setv('maxdt', 1e-14)
setv('mindt', 1e-14)
setv('dt', 1e-14)


T = [ [[[0.0]]] ]
setarray('Temp', T)
run(1e-9)

T = [ [[[10.0]]] ]
setarray('Temp', T)
run(1e-9)

T = [ [[[273.0]]] ]
setarray('Temp', T)
run(1e-9)

T = [ [[[600.0]]] ]
setarray('Temp', T)
run(1e-9)

printstats()

sync()
