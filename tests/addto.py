from mumax2 import *

# Tests the add_to() api

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('micromagnetism')
load('solver/rk12')

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-12)
setv('m_maxerror', 1./1000)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>", "m_error", "dt"], "m.txt", 10e-12)

steps(2)
run(1e-9)

add_to("H_eff", "H_ext")
add_to("H_ext", "H1")
add_to("H_ext", "H2")
add_to("H_ext", "H3")

Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 

setv('H1', [Hx, 0, 0])
setv('H2', [0, Hy, 0])
setv('H3', [0, 0, Hz])

setv('alpha', 0.02)
setv('dt', 0.2e-12)

steps(2)
run(1e-9)

