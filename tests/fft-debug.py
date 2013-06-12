from mumax2 import *

# Standard Problem 4

# define geometry

# number of cells
Nx = 2
Ny = 2
Nz = 1
setgridsize(Nx, Ny, Nz)

# physical size in meters
sizeX = 10e-9
sizeY = 10e-9
sizeZ = 10e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)


# load modules

load('demag')

# set parameters

setv('Msat', 800e3)

# set magnetization
m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

save("B", "gplot", [])
