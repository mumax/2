from mumax2 import *
from mumax2_geom import *
from mumax2_magstate import *

Nx = 128
Ny = 128
Nz = 1

setgridsize(Nx, Ny, Nz)
length=1000e-9
thickness=50e-9
setcellsize(length/Nx, length/Ny, thickness/Nz)
setperiodic(1, 1, 0)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1./3000)

msat=makearray(1, 4, 4, Nz)
msat[0][0][1][0] = 1
msat[0][0][2][0] = 1
msat[0][3][1][0] = 1
msat[0][3][2][0] = 1
setmask('Msat', msat)
#save('Msat', 'omf', ['text'], 'msat.omf')

setarray('m', vortex(1,1))
setcell('m', 0, Ny/2, 0, [0, 0, 1])

autosave('m', 'omf', ['text'], 250e-12)
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * 800e3)

save('m', 'omf', ['Text'], 'vortexX.omf')

