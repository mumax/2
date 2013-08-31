# -*- coding: utf-8 -*-

from math import *
from mumax2_geom import *


Nx = 64
Ny = 64
Nz = 4

sX = 320-9
sY = 320e-9
sZ = 20e-9

hsX = 0.5 * sX
hsY = 0.5 * sY
hsZ = 0.5 * sZ

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)
# setperiodic(8,8,0)

# LLB R
load('exchange6')
load('demag')
load('zeeman')

load('llbar')
load('llbar/torque')
load('llbar/damping/nonconservative/00/local')

loadargs('dissipative-function',[], ["m:mf","R:llbar_local00nc"],["Qmag:Qnc"])

load('temperature/ETM')
load('temperature/ETM/Qspatial')

load('mfa/msat0')
load('mfa/ϰ')

add_to('llbar_RHS', 'llbar_torque')
add_to('llbar_RHS', 'llbar_local00nc')

add_to("Qe", "Qnc")
add_to("Qe", "Qlaser")
add_to("Qe", "Qe_spat")

load('solver/am12')
setv('mf_maxabserror', 1e-4)
setv('mf_maxrelerror', 1e-3)

setv('Te_maxabserror', 1e-4)
setv('Te_maxrelerror', 1e-3)

savegraph("graph.png")

Ms0 = 480e3

T = [ [[[200.0]]] ]
setarray('Te', T)

setv('Tc', 631)
setv('n', 9.14e28)
setv('J', 1.0/2.0)

# Py

msat = [ [[[1.0]]]]
setmask('msat', msat)
setmask('msat0', msat)
setmask('msat0T0', msat)
setmask('ϰ', msat)

mf =[ [[[1.0]]], [[[0.0]]], [[[0.0]]] ]
setarray('Mf', mf)

setv('msat', Ms0)
setv('msat0', Ms0)
setv('msat0T0', Ms0)

Aex = 0.86e-11
setv('Aex', Aex)
setv('γ_LL', 2.211e5)

#~ Bx = 0.0 # 270 Oe
#~ By = 0.0
#~ Bz = 0.0
#~ setv('B_ext',[Bx,By,Bz])

setv('dt', 1e-15)
setv('ϰ', 1e-4)

# Heat bath parameters
setv('k_e', 91.0)

setv('Cp_e', 1070.0)

cpe = makearray(1, Nx, Ny, Nz)
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            val2 = 1.0
            if ii == 0 or ii == Nx - 1:
                    val2 = 0.0
            cpe[0][ii][jj][kk] = val2
setmask('Cp_e', cpe)

# Baryakhtar relaxation parameters

lbd = 0.02
setv('λ∥', [lbd, lbd, lbd])

tt = 1e-15
T0 = 500e-15 # Time delay for the excitation
dT = 80e-15 # FWHM of the excitation gaussian envelope
dTT = 0.5 / (dT * dT) # FWHW squared
Qamp = 1e21
N = 3100 #
time = N * tt
fine = 10
N_fine = fine * N
tt_fine = tt / float(fine)

for i in range(N_fine):
        t = tt_fine * float(i)
        Q = Qamp * exp(-1.0 * dTT * (t-T0)**2)
        setpointwise('Qlaser', t, Q)
setpointwise('Qlaser', 9999.9, 0)

autotabulate(["t", "<msat>"], "msat.txt", tt)
autotabulate(["t", "<Te>"], "Te.txt", tt)
autotabulate(["t", "<msat0>"], "msat0.txt", tt)
autotabulate(["t", "<mf>"], "mf.txt", tt)
autotabulate(["t", "<Qnc>"], "Qnc.txt", tt)
autotabulate(["t", "<Qlaser>"], "Qlaser.txt", tt)

setv('maxdt', 1e-12)
setv('mindt', 1e-17)
setv('dt', 1e-15)

run(10e-12)

printstats()

sync()
