from mumax2 import *
from math import *

# Test for LLB with 2TM
# Ni, just like in PRB 81, 174401 (2010)
  
Nx = 64
Ny = 64
Nz = 4

sX = 640e-9
sY = 640e-9
sZ = 20e-9

hsX = 0.5 * sX
hsY = 0.5 * sY
hsZ = 0.5 * sZ

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)
setperiodic(8,8,0)

# LLB 
load('exchange6')
load('demag')
load('zeeman')
load('llb')
load('dissipative-function')

load('temperature/ETM')
load('temperature/LTM')
load('temperature/E-L')

load('mfa-brillouin/msat0')
load('mfa-brillouin/kappa')

add_to("Qe", "Qmag")
add_to("Qe", "Qlaser")

add_to_weighted("Qe", "Qel", 1.0)
add_to_weighted("Ql", "Qel", -1.0)

load('solver/bdf-euler-auto')
setv('mf_maxiterations', 5)
setv('mf_maxerror', 1e-6)
setv('mf_maxitererror', 1e-8)

setv('Temp_maxiterations', 5)
setv('Temp_maxerror', 1e-6)
setv('Temp_maxitererror', 1e-8)

setv('Te_maxiterations', 5)
setv('Te_maxerror', 1e-6)
setv('Te_maxitererror', 1e-8)

savegraph("graph.png")

Ms0 = 480e3

T = [ [[[200.0]]] ]
setarray('Temp', T)
setarray('Te', T)

setv('Tc', 631)
#~ setv('n', 8.3e28) #Py
setv('n', 9.14e28) #Ni
setv('J', 1.0/2.0)

# Py

Mfd = makearray(3,Nx,Ny,Nz)
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            Mfd[0][ii][jj][kk] = 1.0
            Mfd[1][ii][jj][kk] = 0.0
            Mfd[2][ii][jj][kk] = 0.0
setarray('Mf',Mfd)

msat = makearray(1,Nx,Ny,Nz)            
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat[0][ii][jj][kk] = 1.0

setmask('msat', msat) 
setmask('msat0', msat)
setmask('kappa', msat)

setv('msat', Ms0)        
setv('msat0', Ms0) 
setv('msat0T0', Ms0) 

Aex = 0.86e-11
setv('Aex', Aex)
setv('gamma_LL', 2.211e5)

#~ Bx = 0.0 # 270 Oe
#~ By = 0.0 
#~ Bz = 0.0
#~ setv('B_ext',[Bx,By,Bz])
                
setv('dt', 1e-18)
setv('kappa', 1e-4)

# Heat bath parameters

setv('Cp_e', 3000.0)

setv('Cp_l', 3.1e6)

setv('Gel', 1.0e18)

# Baryakhtar relaxation parameters

lbd = 0.045
lex = Aex / (mu0 * Ms0 * Ms0) 
print("l_ex^2: "+str(lex)+"\n")
lambda_e = 0.0 * lex
setv('lambda', [lbd, lbd, lbd, 0.0, 0.0, 0.0])
setv('lambda_e', [lambda_e, lambda_e, lambda_e, 0.0, 0.0, 0.0])

tt = 1e-15
T0 = 500e-15 # Time delay for the excitation
dT = 80e-15 # FWHM of the excitation gaussian envelope
dTT = 0.5 / (dT * dT) # FWHW squared
Qamp = 3e21
N = 3100 # 
time = N * tt
fine = 10
N_fine = fine * N
tt_fine = tt / float(fine)

for i in range(N_fine):
        t = tt_fine * float(i)
        Q = Qamp * exp(-1.0 * dTT * (t-T0)**2)
        setpointwise('Qlaser', t, Q)  

autotabulate(["t", "<msat>"], "msat.txt", tt)
autotabulate(["t", "<Temp>"], "Tl.txt", tt)
autotabulate(["t", "<Te>"], "Te.txt", tt)
autotabulate(["t", "<msat0>"], "msat0.txt", tt)

autotabulate(["t", "<Qmag>"], "Qmag.txt", tt)
autotabulate(["t", "<Qel>"], "Qel.txt", tt)
autotabulate(["t", "<Qe>"], "Qe.txt", tt)

setv('maxdt', 1e-14)
setv('mindt', 1e-17)
setv('dt', 1e-17)

run(3e-12)

printstats()

sync()
