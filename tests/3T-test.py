from mumax2 import *

# Standard Problem 4

Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(64e-9/Nx, 64e-9/Ny, 2e-9/Nz)

load('3T')
load('solver/rk12')
setv('dt', 1e-15)
setv('Te_maxerror', 1./1000.)
setv('Ts_maxerror', 1./1000.)
setv('Tl_maxerror', 1./1000.)

Te = [ [[[100.0]]] ]
Tl = [ [[[0.0]]] ]
Ts = [ [[[0.0]]] ]
setarray('Te', Te)
setarray('Tl', Tl)
setarray('Ts', Ts)



savegraph("graph.png")

setv('gamma_e', 0.2)
setv('Cs', 0.2)
setv('Cl', 0.2)

setv('Gel', 1.0e9)
setv('Ges', 1.0e9)
setv('Gsl', 1.0e8)

setv('Q', 0.0)

autotabulate(["t", "<Te>", "<Ts>", "<Tl>",], "T.dat", 1e-12)
run(1e-9)

printstats()

sync()
