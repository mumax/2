# -*- coding: utf-8 -*-

from math import *
from mumax2_geom import *


Nx = 64
Ny = 64
Nz = 4

sX = 320e-9
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
setperiodic(4,4,0)

# LLBar
load('exchange6')
load('demag')
# load('zeeman')

load('llbar')
load('llbar/torque')
load('llbar/damping/nonconservative/00/local')
load('llbar/damping/nonconservative/02/local')

load('temperature/ETM')
load('temperature/LTM')
load('temperature/E-L')

load('temperature/sum')

loadargs('mfa/longfield', ["T:Teff"],[],[])
loadargs('mfa/msat0', ["T:Teff"],[],[])
loadargs('mfa/ϰ', ["T:Teff"],[],[])
loadargs('mfa/energy-flow', [], ["mf:mf","R:llbar_local00nc"],["q_s:qs00"])
loadargs('mfa/energy-flow', [], ["mf:mf","R:llbar_local02nc"],["q_s:qs02"])

add_to('llbar_RHS', 'llbar_torque')
add_to('llbar_RHS', 'llbar_local00nc')

add_to_weighted("Qe", "qs00", 1.0)
add_to_weighted("Qe", "qs02", 1.0)

add_to("Qe", "Qlaser")
add_to_weighted("Qe", "Qel", 1.0)
add_to_weighted("Ql", "Qel", -1.0)

load('solver/am12')
setv('mf_maxabserror', 1e-4)
setv('mf_maxrelerror', 1e-4)

setv('Te_maxabserror', 1e-4)
setv('Te_maxrelerror', 1e-4)
setv('Tl_maxabserror', 1e-4)
setv('Tl_maxrelerror', 1e-4)

savegraph("graph.png")

Ms0 = 548e3

T = [ [[[300.0]]] ]
setarray('Te', T)
setarray('Tl', T)

ro = 9.13830273141122913505e28
Tc = 633
J = 0.5

setv('Tc', Tc)
setv('n', ro)
setv('J', J)

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
setv('ϰ', 1.0)

# Heat bath parameters
setv('Cp_e', 654.74383987282318086567) # mean of gamma(T=300) and gamma(T=700)
setv('Cp_l', 3620637.32928679817905918058) # mean of C(T=300) and C(T=Tc)
setv('Gel', 3.5e17) # at 300 K

setv('R', 0.5)

lbd = 0.0045
mu = lbd * 1e-2
setv('λ∥', [lbd, lbd, lbd])
setv('μ∥', [mu, mu, mu])

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
autotabulate(["t", "<Teff>"], "T.txt", tt)
autotabulate(["t", "<msat0>"], "msat0.txt", tt)
autotabulate(["t", "<ϰ>"], "kappa.txt", tt)
autotabulate(["t", "<mf>"], "mf.txt", tt)
autotabulate(["t", "<qs00>"], "qs00.txt", tt)
autotabulate(["t", "<qs02>"], "qs02.txt", tt)
autotabulate(["t", "<Qe>"], "Qe.txt", tt)
autotabulate(["t", "<Qlaser>"], "Qlaser.txt", tt)
autotabulate(["t", "<Qel>"], "Qel.txt", tt)


setv('maxdt', 1e-12)
setv('mindt', 1e-17)
setv('dt', 1e-15)

run(10e-12)

printstats()

sync()
