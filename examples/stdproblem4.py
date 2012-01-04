from mumax2 import *

# Standard Problem 4

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
setv('alpha', 0.02)
setv('dt', 1e-12) # initial time step, will adapt
setv('m_maxerror', 1./1000)


# set magnetization
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


#relax

setv('alpha', 1)    # high damping for relax
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * gets('msat'))
setv('alpha', 0.02) # restore normal damping
setv('t', 0)        # re-set time to 0 so output starts at 0


# schedule some output

autosave("m", "omf", ["Text"], 20e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)


# apply field

# H, as the name implies, is in A/m
Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 
setv('H_ext', [Hx, Hy, Hz])


# run with low damping

setv('alpha', 0.02)
setv('dt', 1e-15) # start over with small time step, will adapt
run(1e-9)


# some debug output

printstats()
savegraph("graph.png")
