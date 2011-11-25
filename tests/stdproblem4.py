from mumax2 import *

# Standard Problem 4


setgridsize(128, 32, 1)
setcellsize(500e-9, 125e-9, 3e-9)

load('micromagnetism')
load('demagexch')

setscalar('Msat', 800e3)
setscalar('Aex', 1.3e-11)
setscalar('alpha', 1)
setscalar('dt', 10e-12)
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


autosave("m", "omf", ["Text"], 200e-12)
autotabulate(["t", "<m>"], "m.txt", 10e-12)
autotabulate(["t", "<H>"], "H.txt", 10e-12)

steps(10000)

Hx = -24.6E-3 / mu0
Hy =   4.3E-3 / mu0
Hz =   0      / mu0 
setvalue('H_ext', [Hx, Hy, Hz])

setscalar('alpha', 0.02)
setscalar('dt', 0.1e-12)

#steps(10000)

printstats()
savegraph("graph.png")

