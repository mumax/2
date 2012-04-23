from mumax2 import *

# Standard Problem 4

# define geometry

# number of cells
Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)

# physical size in meters
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
setv('dt', 1e-15)   # restore time step, will adapt again


# schedule some output

# save magnetization snapshots in OMF text format every 20ps
autosave("m", "omf", ["Text"], 20e-12)
# save a table with time and the average magnetization every 10ps
autotabulate(["t", "<m>"], "m.txt", 10e-12)


# run with field

Bx = -24.6E-3 
By =   4.3E-3 
Bz =   0      
setv('B_ext', [Bx, By, Bz])
run(1e-9)


# some debug output

printstats()
savegraph("graph.png") # see stdprobl4.py.out/graph.dot.png
sync()
