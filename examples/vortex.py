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

# Set material parameters 
setv('Msat', 800e3)   # saturation magnetization
setv('Aex', 1.3e-11)  # exchange coefficient

# Set disk geometry
disk=ellipse()       # returns matrix with 1 inside disk, 0 outside
setmask('Msat', disk)# Msat is multiplied by the disk mask

# Set up solver
load('solver/rk12')  # adaptive Euler-Heun solver
setv('dt', 1e-15)    # inital time step, will adapt
setv('m_maxerror', 1./3000) # maximum error per step

# Set initial magnetization
m = vortex(1,1)   # returns matrix with approximate vortex state 
setarray('m', m) 
setv('alpha', 1)  # relax vortex with high damping
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * 800e3)
setv('t', 0)      # reset time to zero
setv('dt', 1e-15) # reset time step
setv('alpha', 0.01)

# Set up an applied field
N=200
f=1e9 # 1GHz
for i in range(N):
		t=10e-12*i
		B=20e-3 
		Bx=B*sin(2*pi*f*t)
		setpointwise('B_ext', t, [Bx, 0, 0])


# Schedule output:
# Save a table with t and average m every 10ps.
autotabulate(["t", "<m>"], "m.txt", 10e-12)
# Save the full magnetization every 10ps.
autosave("m", "png", [], 50e-12)
autosave("m", "omf", ["Text"], 50e-12)

# run simulation
run(1e-9) 
sync()
