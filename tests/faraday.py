from mumax2 import *

# Test for Faraday's law

Nx = 16 
Ny = 16
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('faraday')
setpointwise('B_ext', 0, [0, 0, 0])
setpointwise('B_ext', 1e-9, [0, 0, 1]) 
setpointwise('B_ext', 2e-9, [0, 0, 0]) 

autotabulate(['B_ext'], 'B_ext.txt', 2e-12)
setv('dt', 1e-12)
run(3e-9)

printstats()
savegraph("graph.png")
