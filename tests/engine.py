from mumax2 import *
from sys import *

# test file for basic engine api

setgridsize(8, 8, 1)
setcellsize(5e-9, 5e-9, 50e-9)

load('micromagnetism')
load('solver/euler')

setscalar('alpha', 0.1)
a=getscalar('alpha')
if a != 0.1:
		exit(-1)

setscalar('Msat', 800e3)
setscalar('Aex', 12e-13)

m=[ [[[0.01]]], [[[0]]], [[[-1]]] ]
setarray('m', m)
Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 

setvalue('H_ext', [Hx, Hy, Hz])

setscalar('dt', 1e-12)
autosave("m", "omf", ["Text"], 10e-12)
autosave("m", "ovf", ["Text"], 10e-12)
autosave("m", "bin", [], 10e-12)
autotabulate(["t", "H_ext"], "h.txt", 10e-12)
autotabulate(["t", "<m.x>"], "mx.txt", 10e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)
steps(100)

printstats()
savegraph("graph.png")

