from mumax2 import *

# Test for LLB
# see I. Radu et al., PRL 102, 117201 (2009)
  
Nx = 32
Ny = 32
Nz = 4

setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)
#setperiodic(8,8,0)

# LLB 
load('exchange6')
load('demag')
load('zeeman')
load('llb')

load('solver/rk12')
setv('m_maxerror', 1./500)
setv('msat_maxerror', 1./500)

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

Bx = 0.0270 # 270 Oe
By = 0.0 
Bz = 0.0
#setv('B_ext',[Bx,By,Bz])

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

           
msat0 = makearray(1,Nx,Ny,Nz)
             
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0) 
setv('msat0', 800e3)

setv('dt', 1e-18)
setv('maxdt',1e-12)
setv('lambda', 1)
setv('kappa', 1e-5) # Chantrell's data
setv('lambda_e', 0.01)

#autotabulate(["t", "<bdl>"], "bdl.txt", 1e-18)
#autotabulate(["t", "<bdt>"], "bdt.txt", 1e-18)
#autotabulate(["t", "<msat>"], "msat.txt", 1e-18)
#autotabulate(["t", "<m>"], "m.txt", 1e-18)

#run(4e-9)
save("m","ovf",[])
save("m","png",[])
save("msat","png",[])

msat = makearray(1,Nx,Ny,Nz) 

for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 0.2

setmask('msat',msat)

setv('dt', 1e-18)
setv('lambda', 0.01)
setv('kappa', 1e-5) # Chantrell's data
setv('lambda_e', 0)

autosave("m", "gplot", [], 10e-15)
autosave("msat", "gplot", [], 10e-15)
#autosave("m", "ovf", [], 10e-15)
#autosave("bdl", "gplot", [], 10e-15)
#autosave("bdl", "png", [], 10e-15)
autotabulate(["t", "<m>"], "m.txt", 1e-16)
autotabulate(["t", "<msat>"], "msat.txt", 1e-16)
#autotabulate(["t", "<bdl>"], "bdl.txt", 10e-15)
#autotabulate(["t", "<bdt>"], "bdt.txt", 10e-15)
#autotabulate(["t", "<H_lf>"], "hlf.txt", 10e-15)

run(100e-12)
#save("bdl","gplot",[])
#save("bdt","gplot",[])
#step()
#save("H_eff","txt",[])
#save("H_eff","gplot", [])
#save("H_lf","gplot",[])
#save("bdl","gplot",[])
#save("bdt","gplot",[])
#save("H_eff","png",[])
printstats()

sync()
