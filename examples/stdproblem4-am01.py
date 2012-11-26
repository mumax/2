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

load('exchange6')
load('demag')
load('zeeman')
load('llg')
load('maxtorque')

load('solver/am01')
setv('m_maxiterations', 3)
setv('m_maxabserror', 1e-4)
setv('m_maxrelerror', 1e-4)
setv('m_maxitererror', 1e-4)
setv('maxdt', 1e-10)
setv('mindt', 1e-17)

# set parameters

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 0.02)
setv('dt', 1e-12) # initial time step, will adapt


# set magnetization
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


#relax

setv('alpha', 1)    # high damping for relax
autotabulate(["t", "maxtorque"], "t.dat", 1e-13)
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * gets('msat'))
setv('alpha', 0.02) # restore normal damping
setv('t', 0)        # re-set time to 0 so output starts at 0
setv('dt', 1e-15)   # restore time step, will adapt again
save("m","vtk",[])

# schedule some output

# save magnetization snapshots in OMF text format every 20ps
# autosave("m", "gplot", [], 1e-12)
# save a table with time and the average magnetization every 10ps
autotabulate(["t", "<m>"], "m.txt", 10e-12)


# run with field

Bx = -24.6E-3 
By =   4.3E-3 
Bz =   0      
setv('B_ext', [Bx, By, Bz])
autotabulate(["t", "m_error"], "error.dat", 1e-13)
run(1e-9)

# some debug output

printstats()
savegraph("graph.png") # see stdprobl4.py.out/graph.dot.png
sync()
