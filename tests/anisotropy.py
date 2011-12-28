from mumax2 import *
from mumax2_material import *
from mumax2_geom import *


Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(200e-9/Nx, 200e-9/Ny, 10e-9/Nz)

load('micromagnetism')
load('anisotropy/uniaxial')
load('solver/rk12')
savegraph('graph.png')

setv('Msat', Co.msat)
setv('Aex', Co.aex)
setv('Ku1', Co.ku1)
setv('anisU', [1, 0, 0])
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1./1000)

setmask('msat', ellipse())

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>", "m_error", "dt"], "m.txt", 10e-12)

run_until_smaller('maxtorque', 1e-3 * gets('gamma') * Co.msat)

