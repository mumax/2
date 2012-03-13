from mumax2 import *
from mumax2_magstate import * # needed for vortex()
from mumax2_geom import * # needed for ellipse()
from math import *

# Vortex gyration

# Set the number of cells.
# power of two is best
Nx = 128
Ny = 128
Nz = 1
setgridsize(Nx, Ny, Nz)

# Set the cell size based on total size
sizeX = 500e-9
sizeY = 500e-9
sizeZ = 50e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)

# Load modules
load('micromagnetism')
load('temperature/brown')
setv('temp', 300)

# Set material parameters 
setv('Msat', 800e3)   # saturation magnetization
setv('Aex', 1.3e-11)  # exchange coefficient
setv('alpha', 0.01)

# Set disk geometry
disk=ellipse()       # returns matrix with 1 inside disk, 0 outside
setmask('Msat', disk)# Msat is multiplied by the disk mask

# Set up solver
load('solver/rk12')  # adaptive Euler-Heun solver
setv('dt', 1e-15)    # inital time step, will adapt
setv('m_maxerror', 1./1000) # maximum error per step

# Set initial magnetization
m = vortex(1,1)   # returns matrix with approximate vortex state 
setarray('m', m) 
setv('t', 0)      # reset time to zero

# Set up an applied field
B = -0.3 # Tesla
setv('B_ext', [0, 0, B])


# Schedule output:
# Save a table with t and average m every 10ps.
autotabulate(["t", "<m>"], "m.txt", 10e-12)
# Save the full magnetization every 10ps.
autosave("m", "omf", ["Text"], 50e-12)

# run simulation
run(2e-12) 
sync()
