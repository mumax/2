# Lets test Zhang-Li against Nmag
# @author Mykola Dvornik

from mumax2 import *
from mumax2_geom import *

Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)

# physical size in meters
sizeX = 100e-9
sizeY = 100e-9
sizeZ = 10e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)

load('micromagnetism')
load('solver/rk12')
load('zhang-li')

savegraph("graph.png")

setv('xi',0.05)
setv('polarisation',1.0)

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1.0)

setv('dt', 1e-15)
setv('maxdt', 1e-12)
setv('m_maxerror', 1./500)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

# Set a initial magnetisation which will relax into a vortex
mValues = {"all":1.0}
InitVortexRegionVectorQuant('m', mValues, [sizeX/2,sizeY/2,0.0], [0.0,0.0,1.0], 1, 1, 0 )

run(1e-9)

setv('alpha', 0.1)


j=makearray(3, 1, 1, 1)
j[0][0][0][0] = 0
j[1][0][0][0] = 0
j[2][0][0][0] = 1e12 # Z component
setarray('j', j)

autosave("m", "png", [], 50e-12)
autotabulate(["t", "<m>"], "m.txt", 50e-12)

run(10.0e-9)

printstats()

quit()