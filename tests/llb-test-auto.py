from mumax2 import *

# Test for LLB
# see I. Radu et al., PRL 102, 117201 (2009)
  
Nx = 32
Ny = 32
Nz = 4

setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)
#setperiodic(16,16,0)

# LLB 
load('exchange6')
load('demag')
load('zeeman')
load('llb')

load('solver/bdf_euler_auto')
setv('mf_maxiterations', 5)
setv('mf_maxerror', 1e-6)
setv('mf_maxitererror', 1e-6)
setv('maxdt', 1e-12)
setv('mindt', 1e-30)

savegraph("graph.png")

Ms0 = 800e3
# Py
Mf = makearray(3,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            Mf[0][ii][jj][kk] = 1.0
            Mf[1][ii][jj][kk] = 0.0
            Mf[2][ii][jj][kk] = 0.0
setarray('Mf',Mf)

msat = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 1.0

setmask('msat', msat) 
setv('msat', Ms0)        

Aex = 1.3e-11
setv('Aex', Aex)
setv('gamma_LL', 2.211e5)

Bx = 0.0270 # 270 Oe
By = 0.0 
Bz = 0.0
setv('B_ext',[Bx,By,Bz])
              
msat0 = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0) 
setv('msat0', Ms0)

setv('dt', 1e-18)
#setv('maxdt',1e-12)
setv('lambda', 0.01)
setv('kappa', 7e-6)
lex = Aex / (mu0 * Ms0 * Ms0) 
print("l_ex^2: "+str(lex)+"\n")
lambda_e = 1e-3 * lex
setv('lambda_e', lambda_e)

Mf = makearray(3,Nx,Ny,Nz) 
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            Mf[0][ii][jj][kk] = 0.2
            Mf[1][ii][jj][kk] = 0.0
            Mf[2][ii][jj][kk] = 0.0
setarray('Mf',Mf)
            
autosave("m", "gplot", [], 1e-12)
autosave("msat", "gplot", [], 1e-12)
#autosave("Mf","gplot", [], 10e-15)
autotabulate(["t", "<m>"], "m.txt", 1e-18)
autotabulate(["t", "badsteps"], "badsteps.txt", 1e-18)
autotabulate(["t", "bdf_iterations"], "i.txt", 1e-18)
autotabulate(["t", "<msat>"], "msat.txt", 1e-18)
#autotabulate(["t", "dt"], "dt.txt", 1e-15)
#autotabulate(["t", "<Mf>"], "Mf.txt", 1e-16)

#step()
run(1e-9)
printstats()

sync()
