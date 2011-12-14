from mumax2 import *
from mumax2_geom import *

Nx = 16
Ny = 16
Nz = 1

setgridsize(Nx, Ny, Nz)
length=500e-9
thickness=20e-9
setcellsize(length/Nx, length/Ny, thickness/Nz)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 0.1e-12)
setv('m_maxerror', 1./1000)

ellipse=ellipsoid(length/2, length/2, float('Inf'))
setmask('Msat', ellipse)
save('Msat', 'omf', ['text'], 'msat.omf')

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

autosave("m", "omf", ["Binary4"], 200e-12)

run(2e-9) #relax

