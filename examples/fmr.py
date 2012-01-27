from mumax2 import *
from math import *

# fmr example
# define geometry

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)

sizeX = 500e-9
sizeY = 125e-9
sizeZ = 3e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)

# set optional PBC here
periodic(0, 0, 0)


# load modules

load('micromagnetism')
load('solver/rk12')


# set parameters

setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 0.01)
setv('dt', 1e-15) # will adapt
setv('m_maxerror', 1./3000)


# set initial magnetization

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

# bias field
staticX = 1/mu0    # 1T static field in X-direction

#relax

setv('alpha', 1)    # high damping for relax
setv('H_ext', [staticX, 0, 0])
run_until_smaller('maxtorque', 1e-3 * gets('gamma') * gets('msat'))
setv('alpha', 0.01) # restore normal damping
setv('t', 0)        # re-set time to 0 so output starts at 0
setv('dt', 1e-15)


# define oscillating field
omega1 = 2*pi*10e9 # frequency1: 1GHz
amplY = 0.1/mu0	   # amplitude in Y-direction
for i in range(1000): # define field with 1000 points
	t = (i/10.)/omega1 # set number of points per period
	setpointwise('H_ext', t, [staticX, amplY*sin(omega1*t), 0])

# schedule output

# save snapshot every 20 ps
autosave("m", "omf", ["Text"], 20e-12)

# save table with time, average m, average field1 and average field2 every 10e-12
# one should check this file to see if the fields are defined as expected
autotabulate(["t", "h_ext", "<m>"], "m.txt", 1e-12)


run(2e-9)

# fmr response: deviation from m along X
my = gets("<m.y>")
mz = gets("<m.z>")
resp = sqrt(my*my + mz*mz)

echo("resp=" + str(resp))
