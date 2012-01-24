from mumax2 import *
from os import *

# Tests maxwell dipole field in 2D

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
cell=1e-9
setcellsize(cell, cell, cell)

load('demag')

