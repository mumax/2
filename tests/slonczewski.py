from mumax2 import *
from mumax2_geom import *

Nx = 64
Ny = 64
Nz = 1

sX = 129.6e-9
sY = 72.0e-9
sZ = 3.0e-9
 
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('solver/rk12')
load('slonczewski')

savegraph("graph.png")

setv('lambda',2.0)
setv('Pol',0.4)
setv('epsilon_prime', 0.1)

setv('Msat', 800e3)
setmask('Msat', ellipse())
setv('Aex', 1.3e-11)
setv('alpha', 0.01)

setv('dt', 1e-15)
setv('maxdt', 1e-12)
setv('m_maxerror', 1./500)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

autosave("m", "png", [], 5e-12)
autotabulate(["t", "<m>"], "m.txt", 1e-12)

run(1e-9)



j=makearray(3, 1, 1, 1)
j[0][0][0][0] = 0
j[1][0][0][0] = 0
j[2][0][0][0] = 1 # Z component
setmask('j', j)
setv('j', [0, 0, 1e8])

p=[ [[[-1]]], [[[0]]], [[[0]]] ]
setarray('p', p)

run(2.0e-9)

printstats()
