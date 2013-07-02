# -*- coding: utf-8 -*-
from mumax2 import *
from math import *
eps = 1e-7
ok = True
# number of cells
Nx = 32
Ny = 32
Nz = 32
setgridsize(Nx, Ny, Nz)

# physical size in meters
sizeX = 32e-9
sizeY = 32e-9
sizeZ = 32e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)


load('micromagnetism')

m = makearray(3, Nx, Ny, Nz)

p1 = 4
p2 = 1
p3 = 1

for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            w = float(ii%p1) / float(p1)
            val = 1.0
            if w >= 0.5:
                val = 0.0
            m[0][ii][jj][kk] = val
            w = float(jj%p2) / float(p2)
            val = 1.0
            if w >= 0.5:
                val = 0.0
            m[1][ii][jj][kk] = val
            w = float(kk%p3) / float(p3)
            val = 1.0
            if w >= 0.5:
                val = 0.0
            m[2][ii][jj][kk] = val


setarray('m', m)
save("m", "gplot", [])
save("fft(m)", "gplot", [])

if ok :
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()
else:
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    sys.exit(1)

