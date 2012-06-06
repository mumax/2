from mumax2 import *
from mumax2_material import *
from sys import exit 
from math import *


Nx = 64
Ny = 64
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 2e-9)

load('exchange6')

setv('Aex', Py.aex)
setv('Msat', Py.msat)

savegraph('graph.png')

setv('Msat', 800e3)

m=makearray(3, 2, 1, 1)
m[0][0][0][0] = 1
m[1][1][0][0] = 1
setarray('m', m)

saveas('H_ex', 'txt', [], 'H_ex.txt')
saveas('m', 'txt', [], 'm.txt')

