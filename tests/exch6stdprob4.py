from mumax2 import *

# Standard Problem 4 with separate exchange/demag calc

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('demag')
load('exchange6')
load('llg')
load('solver/rk12')
savegraph("graph.png")


# fails for a != 1
a=1.
setv('Msat', a*800e3)
msat=makearray(1, 1, 1, 1)
msat[0][0][0][0] = 1./a
setmask('msat', msat)
echo("msat="+str(gets("<msat>")))

setv('Aex', 2*1.3e-11)
aex=makearray(1, 1, 1, 1)
aex[0][0][0][0] = 1./2.
setmask('aex', aex)
setv('alpha', 1)
setv('dt', 1e-15)
setv('m_maxerror', 1./3000)

m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


saveas('kern_dipole', 'gplot', [], 'kern.gplot')
autosave("m", "omf", ["Text"], 200e-12)
autosave("H_ex", "omf", ["Text"], 200e-12)
autosave("H_ex", "gplot", [], 200e-12)
autotabulate(["t", "<m>", "B_ext", "<H_eff>"], "m.txt", 5e-12)

run(2e-9) #relax

m=getv('<m>')
echo("my="+str(m[1])+" want 0.123")

Bx = -24.6E-3
By =   4.3E-3
Bz =   0      
setv('B_ext', [Bx, By, Bz])
setv('alpha', 0.02)
setv('dt', 1e-15)

run(1e-9)

printstats()
