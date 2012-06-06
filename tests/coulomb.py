from mumax2 import *
from os import *

# Tests electrical field in 2D

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
cell=1e-9
setcellsize(cell, cell, cell)

load('coulomb')
load('coulomb')
load('coulomb')

rho = makearray(1, Nx, Ny, Nz)
rho[0][Nx/2][Ny/2][Nz/2] = 1
setarray('rho', rho)

E=getarray('E')
Ehave = E[0][0][Ny/2][0]

r=(Ny/2)*cell
vol=cell**3
q=1*vol
Ewant = -1 * q/(4*pi*epsilon0*r*r)
echo('have ' +  str(Ehave) + ' want '+ str(Ewant))

if abs(Ehave - Ewant) / abs(Ewant) > 1e-4:
	exit(-1)
