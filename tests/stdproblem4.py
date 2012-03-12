from mumax2 import *

# Standard Problem 4

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('micromag/energy')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1./100)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


savegraph("graph.png")

#run(2e-9) #relax
run_until_smaller('maxtorque', 1e-2 * gets('gamma') * 800e3)
setv('alpha', 0.02)
setv('dt', 1e-15)
setv('t', 0)

autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>", "m_error", "m_peakerror", "badsteps", "dt", "maxtorque"], "m.txt", 10e-12)
autotabulate(["t", "E_zeeman"], "Ezeeman.txt", 10e-12)
autotabulate(["t", "E_ex"], "Eex.txt", 10e-12)

Bx = -24.6E-3
By =   4.3E-3
Bz =   0      
setv('B_ext', [Bx, By, Bz])

run(1e-9)

printstats()
