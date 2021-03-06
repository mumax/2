from mumax2 import *
from mumax2_geom import *
from mumax2_magstate import *

Nx = 128
Ny = 128
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
setv('dt', 0.001e-12)
setv('m_maxerror', 1./1000)

msat=ellipsoid(length/2, length/2, float('Inf'))
setmask('Msat', msat)
#save('Msat', 'omf', ['text'], 'msat.omf')

#readarray('m', 'vortex.omf')
setarray('m', vortex(1,1))


autosave('m', 'omf', ['text'], 10e-12)
autosave('m', 'png', [], 10e-12)
run(2e-9)

#save('m', 'omf', [], 'vortex.omf')

