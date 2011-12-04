from mumax2 import *

# Standard Problem 4


Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('demagexch')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-12)
setv('m_maxerror', 1./100)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>", "m_error", "dt"], "m.txt", 10e-12)

run(2e-9) #relax

Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 
setv('H_ext', [Hx, Hy, Hz])
setv('alpha', 0.02)
setv('dt', 0.2e-12)

run(1e-9)

printstats()
savegraph("graph.png")
