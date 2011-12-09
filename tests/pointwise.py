from mumax2 import *

# Standard Problem 4


Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 0.02)
setv('dt', 0.1e-12)
setv('m_maxerror', 1./500)
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)
autotabulate(["t", "<H>"], "H.txt", 10e-12)
autotabulate(["t", "H_ext"], "H_ext.txt", 10e-12)

Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 
setpointwise('H_ext', 0, [0, 0, 0])
setpointwise('H_ext', 1e-9, [0, 0, 0])
setpointwise('H_ext', 1e-9, [Hx, Hy, Hz])
setpointwise('H_ext', 2e-9, [Hx, Hy, Hz])
setpointwise('H_ext', 2e-9, [0, 0, 0])
setpointwise('H_ext', 3e-9, [0, 0, 0])
setpointwise('H_ext', 3e-9, [-Hx, -Hy, -Hz])
setpointwise('H_ext', 4e-9, [-Hx, -Hy, -Hz])
setpointwise('H_ext', 4e-9, [0,0,0])


setv('alpha', 1)
run(1e-9)
setv('alpha', 0.02)
run(5e-9)


printstats()
savegraph("graph.png")

