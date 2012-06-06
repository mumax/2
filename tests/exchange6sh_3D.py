from mumax2 import *

# Standard Problem 4

Nx = 128*2
Ny = 32*2
Nz = 1*2
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

#load('micromagnetism')

#setv('Msat', 800e3)
#setv('Aex', 1.3e-11)
#setv('alpha', 1)
#setv('dt', 1e-15)
#setv('m_maxerror', 1./100)


load('exchange6')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)

setarray_file('m', 'input3D.omf')

savegraph("graph.png")
saveas('H_ex', 'txt', [], 'H_ex.txt')
saveas('m', 'txt', [], 'm.txt')
