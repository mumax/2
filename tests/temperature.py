from mumax2 import *


Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('solver/rk12')
load('temperature/brown')
savegraph("graph.png")

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 0.01)
setv('dt', 1e-12)
setv('m_maxerror', 1./100)
setv('temp', 100)
setmask('temp', [ [[[0], [1]]] ])
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 2e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)
autotabulate(["t", "<H_therm>"], "H_therm.txt", 10e-12)

run(1e-9)

printstats()

