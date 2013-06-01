from mumax2 import *
from math import *
from mumax2_geom import *

# Test for LLB with 2TM
# Ni, just like in PRB 81, 174401 (2010)
  
Nx = 64
Ny = 64
Nz = 4

sX = 320e-9
sY = 320e-9
sZ = 20e-9

hsX = 0.5 * sX
hsY = 0.5 * sY
hsZ = 0.5 * sZ

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)
# setperiodic(8,8,0)

# LLB R
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

add_to_weighted("Ql", "Qc", -1.0)

load('solver/rk12')
setv('mf_maxerror', 1e-4)

setv('Temp_maxerror', 1e-3)

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
#~ setv('kappa', 1e-4)

#~ Bx = 0.1 # 1000 Oe
#~ By = 0.0 
#~ Bz = 0.0
#~ setv('B_ext',[Bx,By,Bz])
                
# Heat bath parameters

setv('Cp_l', 3.0e6)

# Baryakhtar relaxation parameters

mu = 0.005
setv('mu', [mu, mu, mu, 0.0, 0.0, 0.0])

tt = 1e-15
 
autotabulate(["t", "<Temp>"], "Temp.txt", tt)
autotabulate(["t", "<mf>"], "mf.txt", tt)
autotabulate(["t", "<Qc>"], "Qc.txt", tt)
autotabulate(["t", "<Ql>"], "Ql.txt", tt)
autotabulate(["t", "<H_therm>"], "H_therm.txt", tt)

setv('maxdt', 1e-15)
setv('mindt', 1e-15)
setv('dt', 1e-15)


T = [ [[[273.0]]] ]
setarray('Temp', T)

run(1e-9)

#~ T = [ [[[600.0]]] ]
#~ setarray('Temp', T)

run(1e-9)

printstats()

sync()
