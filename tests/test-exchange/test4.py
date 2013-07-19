# -*- coding: utf-8 -*-

from mumax2 import *
from random import *
# Standard Problem 4

# define geometry

eps = 1.0e-6

# number of cells
Nx = 64
Ny = 64
Nz = 64
setgridsize(Nx, Ny, Nz)

# physical size in meters
sizeX = 320e-9
sizeY = 160e-9
sizeZ = 64e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)

seed(0)

# load modules

load('exchange6')

# set parameters
msk=makearray(1, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if i > Nx / 2 :
                msk[0][i][j][k] = 1.0
            else :
                msk[0][i][j][k] = 0.5
setmask('Msat', msk)
setv('Msat', 800e3)

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if i > Nx / 2 :
                msk[0][i][j][k] = 1.0
            else :
                msk[0][i][j][k] = 0.5
setmask('Aex', msk)
setv('Aex', 1.3e-11)

# set magnetization
m=makearray(3, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if i > Nx / 2 :
                m[0][i][j][k] = 1.0
                m[1][i][j][k] = 1.0
                m[2][i][j][k] = 0.0
            else :
                m[0][i][j][k] = -1.0
                m[1][i][j][k] = 1.0
                m[2][i][j][k] = 0.0
setarray('m', m)

saveas('H_ex', "dump", [], "hex_test4.dump")

