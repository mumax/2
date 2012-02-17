from mumax2 import *

# Standard Problem 4

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1./3000)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>", "m_error", "dt", "maxtorque"], "m.txt", 10e-12)

save('kern_dipole.xx', 'gplot', [])
save('kern_dipole.xy', 'gplot', [])
save('kern_dipole.xz', 'gplot', [])
save('kern_dipole.yx', 'gplot', [])
save('kern_dipole.yy', 'gplot', [])
save('kern_dipole.yz', 'gplot', [])
save('kern_dipole.zx', 'gplot', [])
save('kern_dipole.zy', 'gplot', [])
save('kern_dipole.zz', 'gplot', [])

run(2e-9) #relax

Bx = -24.6E-3
By =   4.3E-3
Bz =   0      
setv('B_ext', [Bx, By, Bz])
setv('alpha', 0.02)
setv('dt', 1e-15)

run(1e-9)

printstats()
savegraph("graph.png")
