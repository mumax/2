from mumax2 import *

# Standard Problem 4


Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
setv('mindt', 1e-12)
setv('maxdt', 1e-12)

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 0.02)
setv('dt', 0.5e-12)
#setv('m_maxerror', 1./3000)
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)
autotabulate(["t", "B_ext"], "B_ext.txt", 10e-13)

Bx = -24.6E-3
By =   4.3E-3
Bz =   0      
setpointwise('B_ext', 0, [0, 0, 0])
setpointwise('B_ext', 1e-11, [0, 0, 0])
setpointwise('B_ext', 1e-11, [Bx, By, Bz])
setpointwise('B_ext', 2e-11, [Bx, By, Bz])
setpointwise('B_ext', 2e-11, [0, 0, 0])
setpointwise('B_ext', 3e-11, [0, 0, 0])
setpointwise('B_ext', 3e-11, [-Bx, -By, -Bz])
setpointwise('B_ext', 4e-11, [-Bx, -By, -Bz])
setpointwise('B_ext', 4e-11, [0,0,0])
setpointwise('B_ext', 99999, [0,0,0]) # stay at zero for about forever.

setpointwise('alpha', 0, 1)
setpointwise('alpha', 1e-9, 0.02)

savegraph("graph.png")

setv('alpha', 1)
run(1e-11)
setv('alpha', 0.02)
run(6e-11)


printstats()

