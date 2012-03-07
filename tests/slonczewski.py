from mumax2 import *
from mumax2_geom import *

Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('solver/rk12')
load('slonczewski')
savegraph("graph.png")

setv('Msat', 800e3)
setmask('Msat', ellipse())
setv('Aex', 1.3e-11)
setv('alpha', 0.01)
setv('dt', 0.5e-12)
setv('m_maxerror', 1./100)
setv('aj',1.0)
setv('bj',0.3)
setv('Pol',0.56)

j=makearray(3, 1, 1, 1)
j[0][0][0][0] = 0
j[1][0][0][0] = 0
j[2][0][0][0] = 1e8 # Z component
setarray('j', j)

m=[ [[[1]]], [[[1]]], [[[0]]] ]


p=[ [[[-1]]], [[[0]]], [[[0]]] ]
setarray('m', m)
setarray('p', p)

autosave("m", "png", [], 25e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)

run(2.0e-9)

printstats()
