# Micromagnetics standard proplem no. 5
# As proposed by M. Najafi et al., JAP 105, 113914 (2009).
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

csX = (sizeX/Nx)
csY = (sizeY/Ny)
csZ = (sizeZ/Nz)

setcellsize(csX, csY, csZ)

load('micromagnetism')
load('solver/rk12')

savegraph("graph.png")

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1.0)

setv('dt', 1e-15)
setv('maxdt', 1e-12)
setv('m_maxerror', 1./500)


# Set a initial magnetisation which will relax into a vortex
mv = makearray(3, Nx, Ny, Nz)

for m in range(Nx):
    for n in range(Ny):
        for o in range(Nz):
		
            xx = float(m) * csX - 50.0e-9
            yy = 50.0e-9 - float(n) * csY
			
            mv[0][m][n][o] = yy
            mv[1][m][n][o] = xx
            mv[2][m][n][o] = 40.0e-9

setarray('m', mv)

run_until_smaller('maxtorque', 1e-6 * gets('gamma') * 800e3)

load('zhang-li')

setv('alpha', 0.1)
setv('dt', 1e-15)
setv('t', 0)

tabulate(["t", "<m>"], "m.txt")

setv('xi',0.05)
setv('polarisation',1.0)
save("m","png",[])

j = makearray(3, Nx, Ny, Nz)
for m in range(Nx):
    for n in range(Ny):
        for o in range(Nz):
            j[0][m][n][o] = 1.0
            j[1][m][n][o] = 0.0
            j[2][m][n][o] = 0.0

setv('j', [1e12, 0, 0])
setmask('j', j)

autosave("m", "png", [], 50e-12)
autotabulate(["t", "<m>"], "m.txt", 50e-12)

run(10.0e-9)

printstats()

quit()