from mumax2 import *
from mumax2_geom import *
from mumax2_magstate import *
from math import * 

Nx = 1024
Ny = 4
Nz = 4

setgridsize(Nx, Ny, Nz)
sX = 2048e-9
sY = 8e-9
sZ = 8e-9
setcellsize(sX/Nx, sY/Ny, sZ/Nz)

# LLG 
load('micromagnetism')
load('solver/rk12')

# Py
setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1.0)

setv('dt', 1e-15)
setv('m_maxerror', 1e-4)

# The applied field is strong enough to saturate 
m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

# static applied field
Bx = 1.0 
By = 0.0 
Bz = 0.0 

setv('B_ext', [Bx, By, Bz])

run_until_smaller('maxtorque', 1e-4 * gets('gamma') * 800e3)

setv('alpha', 0.0008)
getdispersion(50e9, 100e9, 5, 0)
