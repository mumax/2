from mumax2 import *

# Test for LLB

Nx = 32 
Ny = 32
Nz = 8
setgridsize(Nx, Ny, Nz)
setcellsize(1e-9, 1e-9, 1e-9)

load('exchange6')
load('demag')
load('zeeman')
load('solver/euler')
load('llb')

# Py
msat = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 1.0

setmask('msat', msat)   
setv('Msat', 800e3)         
setv('Aex', 1.3e-11)
setv('gamma_LL', 2.211e5)
setv('dt', 1e-13)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

#run(1e-9)

msat0 = makearray(1,Nx,Ny,Nz)               
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0) 
setv('msat0', 800e3)
msat = makearray(1,Nx,Ny,Nz)            

for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 0.2

setmask('msat',msat)
setv('dt', 1e-16)
setv('lambda', 8e3)
setv('kappa', 5e-4)
setv('lambda_e', 0.0)

autosave("m", "png", [], 10e-15)
autosave("m", "ovf", [], 10e-15)
autosave("bdl", "gplot", [], 10e-15)
autosave("bdl", "png", [], 10e-15)
#autotabulate(["t", "<m>"], "m.txt", 10e-15)
autotabulate(["t", "<msat>"], "msat.txt", 10e-15)
#autotabulate(["t", "<bdl>"], "bdl.txt", 10e-15)
#autotabulate(["t", "<bdt>"], "bdt.txt", 10e-15)
#autotabulate(["t", "<H_lf>"], "hlf.txt", 10e-15)

run(25e-12)
save("bdl","gplot",[])
#step()
save("H_eff","txt",[])
save("H_eff","gplot", [])
save("H_lf","gplot",[])
save("bdl","gplot",[])
#save("H_eff","png",[])
printstats()

sync()
