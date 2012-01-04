from mumax2 import *

# Example of excitation with 2 localized magnetic fields
# We (ab)use h_bias, which is intended as a additional field for biasing,
# but can be used as a general external field field.


# define geometry

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)

sizeX = 500e-9
sizeY = 125e-9
sizeZ = 3e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)


# load modules

load('micromagnetism')
load('solver/rk12')


# set parameters

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('dt', 1e-12) # will adapt
setv('m_maxerror', 1./3000)


# set initial magnetization

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


#relax

setv('alpha', 1)    # high damping for relax
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * gets('msat'))
setv('alpha', 0.02) # restore normal damping
setv('t', 0)        # re-set time to 0 so output starts at 0
setv('dt', 0.2e-12)

# schedule output
autosave("m", "omf", ["Text"], 20e-12)

Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 
setv('H_ext', [Hx, Hy, Hz])
setv('alpha', 0.02)

run(1e-9)

printstats()
savegraph("graph.png")
