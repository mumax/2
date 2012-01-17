from mumax2 import *

# Electrical current paths

Nx = 8
Ny = 8
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('current')

rho = makearray(1, Nx, Ny, Nz)
rho[0][0][0][0] = 1
setarray('rho', rho)


save('kern_el', 'gplot', [], 'kern_el.gplot')
save('rho', 'gplot', [], 'rho.gplot')
save('E', 'gplot', [], 'E.gplot')

printstats()
savegraph("graph.png")
