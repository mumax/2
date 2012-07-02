from mumax2 import *
from math import *
# Test for LLB
# see I. Radu et al., PRL 102, 117201 (2009)
  
Nx = 128
Ny = 128
Nz = 4

sX = 256e-9
sY = 256e-9
sZ = 10e-9

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)
setperiodic(8,8,0)

#optical spot size
osX = 0.05e-6
osY = 0.05e-6

cX = sX/2.0
cY = sY/2.0

sigmaX = 1.0/(2.0*osX*osX)
sigmaY = 1.0/(2.0*osY*osY)

pre = 1.0 / (2.0 * pi * osX * osY)
sdepth = 25e-9
# Wrap into hi-damping regions

wR = 200e-9
cwR = wR * 0.5
max_Lambda = 5.0
Lambda = 0.008
Lambda_slope = 0.5 * (max_Lambda - Lambda)/(sX - wR)

# LLB 
load('exchange6')
load('demag')
load('zeeman')
load('llb')
load('abc_gilbert')

load('solver/bdf_euler_auto')
setv('mf_maxiterations', 5)
setv('mf_maxerror', 1e-6)
setv('mf_maxitererror', 1e-8)
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

msat0 = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    zz = float(kk) * csZ
    for jj in range(Ny):
        yy = float(jj)*csY-cY
        y = yy**2
        for ii in range(Nx):
            xx = float(ii)*csX-cX
            x = xx**2
            arg = -(0.25*x*sigmaX + 0.25*y*sigmaY)
            sd = -zz/sdepth
            scale = 1.0 - 0.1 * exp(arg) * exp(sd) 
            msat0[0][ii][jj][kk] = scale
setmask('msat0', msat0)   
setv('msat0', Ms0)  

Aex = 1.3e-11
setv('Aex', Aex)
setv('gamma_LL', 2.211e5)

Bx = 0.0270 # 270 Oe
By = 0.0 
Bz = 0.0
setv('B_ext',[Bx,By,Bz])
              
setv('dt', 1e-18)
#setv('maxdt',1e-12)
setv('lambda', 0.008)
setv('kappa', 2e-4)
lex = Aex / (mu0 * Ms0 * Ms0) 
print("l_ex^2: "+str(lex)+"\n")
lambda_e = 1e-4 * lex
setv('lambda_e', lambda_e)

Mf = getarray('Mf') 
Mfd = makearray(3,Nx,Ny,Nz)
for kk in range(Nz):
    zz = float(kk) * csZ
    for jj in range(Ny):
        yy = float(jj)*csY-cY
        y = yy**2
        for ii in range(Nx):
            xx = float(ii)*csX-cX
            x = xx**2
            arg = -(x*sigmaX + y*sigmaY)
            sd = -zz / sdepth
            scale = 1.0 - 0.9 * exp(arg) * exp(sd)
            Mfd[0][ii][jj][kk] = Mf[0][ii][jj][kk] * scale
            Mfd[1][ii][jj][kk] = Mf[1][ii][jj][kk] * scale
            Mfd[2][ii][jj][kk] = Mf[2][ii][jj][kk] * scale
setarray('Mf',Mfd)
     
autosave("m", "gplot", [], 1e-12)
autosave("m", "gplot.gz", [], 1e-12)
autosave("m", "gplot.zip", [], 1e-12)

#autosave("msat", "gplot", [], 1e-12)
#autosave("mf", "gplot", [], 1e-12)
#autotabulate(["t", "<m>"], "m.txt", 1e-16)
#autotabulate(["t", "badsteps"], "badsteps.txt", 1e-16)
#autotabulate(["t", "bdf_iterations"], "i.txt", 1e-16)
#autotabulate(["t", "<msat>"], "msat.txt", 1e-16)
#autotabulate(["t", "<mf>"], "mf.txt", 1e-16)
#step()
#save("b", "gplot", [])
run(3e-12)
printstats()

sync()
