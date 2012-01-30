from mumax2 import *
from mumax2_material import *
from sys import exit 
from math import *


Nx = 64
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 2e-9)

load('exchange6')
load('llg')
load('solver/rk12')

setv('Aex', Py.aex)
setv('Msat', Py.msat)
setv('alpha', 0.5)
setv('dt', 1e-15)
setv('m_maxerror', 1./1000)

savegraph('graph.png')

setv('Msat', 800e3)

m=makearray(3, 2, 1, 1)
m[0][0][0][0] = 1
m[1][1][0][0] = 1
setarray('m', m)

save('H_ex', 'gplot', [], 'H_ex.gplot')

autosave('m', 'omf', ['Text'], 10e-12)
run(1e-9)
