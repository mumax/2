from mumax2 import *
from math import *

# Test for 1TM module                               

Nx = 32
Ny = 32
Nz = 1

setgridsize(Nx, Ny, Nz)
setcellsize(64e-9/Nx, 64e-9/Ny, 2e-9/Nz)

load('temperature/ETM')
load('temperature/ETM/Qspatial')
load('temperature/LTM')
load('temperature/LTM/Qspatial')
load('temperature/E-L')

add_to("Qe", "Qe_spat")
add_to("Ql", "Ql_spat")
add_to_weighted("Qe", "Qel", 1.0)
add_to_weighted("Ql", "Qel", -1.0)

load('solver/rk12')
setv('dt', 1e-15)
setv('maxdt', 1e-12)
setv('mindt', 1e-18)

setv('Te_maxerror', 1./1000.)
setv('Te_maxerror', 1./1000.)
setv('Temp_maxerror', 1./1000.)
setv('Temp_maxerror', 1./1000.)

savegraph("graph.png")

setv('k_e', 1.0e3)
setv('Cp_e', 2.0e5)

setv('k_l', 1.0e2)
setv('Cp_l', 1.0e6)

setv('Gel', 0.6e18)

T = [[[[0.0]]]]
setarray('Te',  T)
setarray('Temp', T)


add_to("Qe", "Q")
tt = 1e-15
T0 = 50e-15 # Time delay for the excitation
dT = 10e-15 # FWHM of the excitation gaussian envelope
dTT = 0.5 / (dT * dT) # FWHW squared
Qamp = 0.7e22
N = 2100 # 
time = N * tt
fine = 10
N_fine = fine * N
tt_fine = tt / float(fine)

for i in range(N_fine):
        t = tt_fine * float(i)
        Q = Qamp * exp(-1.0 * dTT * (t-T0)**2)
        setpointwise('Q', t, Q)
        


autotabulate(["t", "<Te>"], "Te.dat", tt)
autotabulate(["t", "<Temp>"], "Tl.dat", tt)
autotabulate(["t", "<Qel>"], "Qel.dat", tt)

autosave("Te", "png", [], tt)
autosave("Temp", "png", [], tt)

run(2e-12)

printstats()

sync()

