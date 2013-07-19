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
setperiodic(1,1,1)

seed(0)

# load modules

load('exchange6')

# set parameters
msk=makearray(1, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            msk[0][i][j][k] = random()
setmask('Msat', msk)
setv('Msat', 800e3)

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            msk[0][i][j][k] = random()
setv('Aex', 1.3e-11)

# set magnetization
m=makearray(3, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            m[0][i][j][k] = random()
            m[1][i][j][k] = random()
            m[2][i][j][k] = random()
setarray('m', m)

saveas('H_ex', "omf", ["Text"], "hex_new_pbc.omf")
ref = readfile(outputdirectory()+"/../hex_ref_pbc.omf")
new = getarray('H_ex')

dirty = 0

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            for c in range(3):
               diff = abs(new[c][i][j][k] - ref[c][i][j][k])
               if diff > eps:
                   dirty = dirty + 1
                   print "error: ",diff, " at ", c, i, j, k

if dirty > 0:
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    sys.exit(1)
else:
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()
