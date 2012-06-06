from mumax2 import *

# Test for LLB

Nx = 32 
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('llg')
load('exchange6')
load('demag')
load('zeeman')
load('solver/euler')

# Py
msat = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 1.0

setmask('msat', msat)            
setv('Msat', 800e3)
setv('Aex', 1.3e-11)
setv('alpha', 1)
setv('gamma', 2.211e5)
setv('dt', 1e-13)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

run(5e-9)

load('llb')
msat0 = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0) 
setv('msat0', 800e3)
setv('dt', 1e-15)
setv('beta', 0.02)
setv('kappa', 1000.0)
setv('blambda', 0.02)

setv('Msat',200e3)

autosave("m", "png", [], 10e-15)
autosave("m", "ovf", [], 10e-15)
autotabulate(["t", "<m>"], "m.txt", 10e-15)
autotabulate(["t", "<msat>"], "msat.txt", 10e-15)
autotabulate(["t", "<bld>"], "msat.txt", 10e-15)
autotabulate(["t", "<blt>"], "msat.txt", 10e-15)
run(100e-12)

printstats()

sync()
