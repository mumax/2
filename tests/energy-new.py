# -*- coding: utf-8 -*-

from mumax2 import *
from math import *

eps = 1.0e-6

Nx = 128
Ny = 32
Nz = 4

hNx = Nx / 2
hNy = Ny / 2
hNz = Nz / 2
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 64e-9/Ny, 10e-9/Nz)

load('zeeman')
load('micromag/energy')

MsA = 800e3
MsB = 0.0001e3
rMs = MsB / MsA

setv('msat', MsA)
setmask('msat', [[[[1.0]]]])
Bx = 1.0
By = 0.0
Bz = 0.0      
setv('B_ext', [Bx, By, Bz])

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

E1 = gets('E_zeeman')

msat=makearray(1, Nx, Ny, Nz)
for i in range(Nx):
    val = 1.0
    if i < hNx:
        val = rMs
    for j in range(Ny):
        for k in range(Nz):
            msat[0][i][j][k] = val
setmask('msat', msat)

E2 = gets('E_zeeman')

ref_rE = (0.5 * 1.0 + 0.5 * rMs)

rE = E2 / E1

drE = abs(rE - ref_rE)

if drE > eps:
    print "\033[31m" + "Exided target precission: " + str(drE) + "(" + str(eps) + ")" + "\033[0m"
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    sys.exit(1)
else:
    print "\033[32m" + "Meet target precission: " + str(drE) + "(" + str(eps) + ")" + "\033[0m"
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()
