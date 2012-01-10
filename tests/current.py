from mumax2 import *

# Electrical current paths

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('current')
printstats()
savegraph("graph.png")