from mumax2 import *

# Test for Faraday's law

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('maxwell')



printstats()
savegraph("graph.png")
