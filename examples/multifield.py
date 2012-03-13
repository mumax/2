from mumax2 import *
from math import *

# Example of excitation with 2 localized magnetic fields
# We use add_to(), which can be used to add new contributions
# to a quantity

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

# MULTI-FIELD EXAMPLE:
# Here we introduce a new quantity B_ext2, to be added to B
# Infinitely many can be added.
add_to('B', 'B_ext2')

# define field1
# mask is thin line at X=10 cells
mask1 = makearray(3, Nx, 1, Nz) # 3 x Nx x Ny x Nz array
mask1[0][10][0][0] = 1 # x-component
mask1[1][10][0][0] = 1 # y-component
mask1[2][10][0][0] = 1 # z-component
setmask('B_ext', mask1)
# masks can also be read from .omf files (readmask, 'B_ext', 'mask.omf')

# define oscillating field
omega1 = 2*pi*10e9 # frequency1: 1GHz
B1x = 0 #T
B1y = 0.1 #T
B1z = 0 #T
for i in range(1000): # 1000 points in total
	t = (i/10.)/omega1 # about 30 points per period
	setpointwise('B_ext', t, [B1x*sin(omega1*t)/mu0, B1y/mu0*sin(omega1*t), B1z/mu0*sin(omega1*t)])


# define field2
# mask is thin line at X=120 cells
mask2 = makearray(3, Nx, 1, Nz) # 3 x Nx x Ny x Nz array
mask2[0][120][0][0] = 1 # x-component
mask2[1][120][0][0] = 1 # y-component
mask2[2][120][0][0] = 1 # z-component
setmask('B_ext2', mask2)

# define oscillating field
omega2 = 2*pi*20e9 # frequency1: 2GHz
B2x = 0 #T
B2y = 0 #T
B2z = 0.1 #T
for i in range(2000):
	t = (i/10.)/omega2
	setpointwise('B_ext2', t, [B2x*sin(omega2*t)/mu0, B2y/mu0*sin(omega2*t), B2z/mu0*sin(omega2*t)])

B2x = 0 #T
B2y = 0 #T
B2z = 0.1 #T
setv('B_ext2', [B2x/mu0, B2y/mu0, B2z/mu0])


# schedule output

# save snapshot every 20 ps
autosave("m", "omf", ["Text"], 20e-12)

# save table with time, average m, average field1 and average field2 every 10e-12
# one should check this file to see if the fields are defined as expected
autotabulate(["t", "<m>", "<B_ext>", "<B_ext2>"], "m.txt", 1e-12)

run(0.1e-9)
sync()
