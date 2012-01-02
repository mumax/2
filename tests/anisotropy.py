# mx -> 0.62

from mumax2 import *
from mumax2_geom import *


Nx = 64
Ny = 64
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(128e-9/Nx, 128e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('anisotropy/uniaxial')
load('solver/rk12')
savegraph('graph.png')

setv('Msat', 800e3)
setmask('Msat', ellipse())
setv('Aex', 13e-12)
setv('Ku1', -500)
setv('anisU', [1, 0, 0])
setv('alpha', 0.25)

setv('dt', 1e-15)
setv('m_maxerror', 1./1000)


m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

autotabulate(["t", "<m>", "<H_anis>"], "m.txt", 10e-12)

run(2e-9)
setv('h_ext', [0, 1e-3/mu0, 0])
run(2e-9)
