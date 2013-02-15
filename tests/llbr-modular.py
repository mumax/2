from mumax2 import *
from math import *
# Test for modular LLBr
  
Nx = 32
Ny = 32
Nz = 32

sX = 256e-9
sY = 256e-9
sZ = 256e-9

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)

load('llbr')
load('llbr/torque')
load('llbr/longitudinal')
load('llbr/transverse')
load('llbr/nonlocal')

savegraph("graph.png")

printstats()

sync()
