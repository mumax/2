from mumax2 import *
from math import *
# Standard Problem 4

Nx = 32
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(64e-9/Nx, 64e-9/Ny, 2e-9/Nz)

load('3T')
load('solver/rk12')
setv('dt', 1e-15)
setv('Te_maxerror', 1./1000.)
setv('Ts_maxerror', 1./1000.)
setv('Tl_maxerror', 1./1000.)
#load('solver/euler')
#setv('dt', 1e-15)

Te = [ [[[10.0]]] ]
Tl = [ [[[10.0]]] ]
Ts = [ [[[10.0]]] ]
setarray('Te', Te)
setarray('Tl', Tl)
setarray('Ts', Ts)



savegraph("graph.png")

setv('gamma_e', 1074.0)
setv('Cs', 1.0e3)
setv('Cl', 12.0e5)

setv('Gel', 8.e17)
setv('Ges', 12.0e15)
setv('Gsl', 8.64e15)

tt = 1e-15 # 1/2*tt = bandwidth ~50 GHz

# Set up an rf applied field
T0 = 100e-15 # Time delay for the excitation
dT = 10e-15 # FWHM of the excitation gaussian envelope
dTT = 0.5 / (dT * dT) # FWHW squared
Qamp = 1e22
N = 1100 # 2100 timesteps, ~6 ns
time = N * tt
fine = 10
N_fine = fine * N
tt_fine = tt / float(fine)

for i in range(N_fine):
        t = tt_fine * float(i)
        Q = Qamp * exp(-1.0 * dTT * (t-T0)**2)
        setpointwise('Q', t, Q)
            
autotabulate(["t", "<Te>", "<Tl>", "<Ts>",], "T.dat", tt)
autotabulate(["t", "Q",], "Q.dat", tt)
autotabulate(["t", "<Qe>", "<Ql>", "<Qs>"], "QQ.dat", tt)
run(1e-12)

printstats()

sync()
