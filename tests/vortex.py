from mumax2 import *

Nx = 128
Ny = 128
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 500e-9/Ny, 20e-9/Nz)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 0.1e-12)
setv('m_maxerror', 1./1000)

msat=[ [[ [1], [0]]] ]
setmask('Msat', msat)
save('Msat', 'omf', ['text'], 'msat.omf')

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

autosave("m", "omf", ["Binary4"], 200e-12)

run(2e-9) #relax

