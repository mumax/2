from mumax2 import *
from math import *
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

# LLB 
load('exchange6')
load('demag')
load('zeeman')
load('llb')

#load('solver/bdf-euler-auto')
#setv('mf_maxiterations', 5)
#setv('mf_maxerror', 1e-6)
#setv('mf_maxitererror', 1e-8)

load('solver/rk12')
setv('mf_maxerror', 1e-6)
setv('mindt', 1e-16)
setv('maxdt', 1e-12)
# set parameters
# Py
# set magnetization
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)

msat = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 1.0

setmask('msat', msat)   
setv('Msat', 800e3) 

Mf = makearray(3,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            Mf[0][ii][jj][kk] = 1.0/sqrt(2.0)
            Mf[1][ii][jj][kk] = 1.0/sqrt(2.0)
            Mf[2][ii][jj][kk] = 0.0
setarray('Mf',Mf)

msat0 = makearray(1,Nx,Ny,Nz)
             
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0) 
setv('msat0', 800e3)

Ms0 = 800e3
Aex = 1.3e-11
lex = Aex / (mu0 * Ms0 * Ms0) 
print("l_ex^2: "+str(lex)+"\n")
lambda_e = 0.0 * lex
setv('lambda_e', lambda_e)
setv('kappa', 1e-4)
setv('Aex', 1.3e-11)
gamma=2.211e5
alpha=1.0
gammall = gamma / (1.0+alpha**2)
setv('gamma_ll',gammall)
setv('dt', 1e-17) # initial time step, will adapt

#relax

setv('lambda', 0.1) # high damping for relax
autotabulate(["t", "<m>"], "m.txt", 1e-12)
run(1e-9)

alpha=0.02
gammall = gamma / (1.0+alpha**2)
setv('gamma_ll',gammall)
setv('lambda', 0.02) # restore normal damping
#setv('t', 0)        # re-set time to 0 so output starts at 0
save("m","vtk",[])
save("m","ovf",[])

# schedule some output

# save magnetization snapshots in OMF text format every 20ps
autosave("m", "gplot", [], 1e-12)
# save a table with time and the average magnetization every 10ps
autotabulate(["t", "<m>"], "m.txt", 1e-12)


# run with field

Bx = -24.6E-3 
By =   4.3E-3 
Bz =   0      
setv('B_ext', [Bx, By, Bz])
setv('dt', 1e-15) # initial time step, will adapt
run(1e-9)


# some debug output

printstats()
savegraph("graph.png") # see stdprobl4.py.out/graph.dot.png
sync()
