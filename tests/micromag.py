from mumax2 import *

# test file for basic micromagnetism

setgridsize(16, 8, 2)

setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')

setscalar('alpha', 0.01)
setscalar('msat', 800e3)
setscalar('aex', 12e-13)

m=[ [[[0.01]]], [[[0]]], [[[-1]]] ]
setarray('m', m)

Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 

setvalue('h_ext', [Hx, Hy, Hz])

setscalar('dt', 1e-12)
autosave("m", "omf", ["Text"], 10e-12)
autosave("m", "ovf", ["Text"], 10e-12)
autosave("m", "bin", [], 10e-12)
autotabulate(["t", "H_ext"], "t.txt", 10e-12)
steps(100)

printstats()
savegraph("graph.png")

