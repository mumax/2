from mumax2 import *
from mumax2_cmp import *

setgridsize(40, 40, 2)
setcellsize(5e-9, 5e-9, 50e-9)

load('micromagnetism')
savegraph("graph.dot")

setscalar('alpha', 0.1)
getscalar('alpha')
setscalar('Msat', 800e3)
setscalar('Aex', 12e-13)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)
setVortex( 'm', (1e-7,1e-7,5e-8), (0.,0.,1.), 1, 1 )
Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 

setvalue('H_ext', [Hx, Hy, Hz])

setscalar('dt', 1e-12)
save("m", "omf", ["Text"], "vortex_with_updater.omf" )
#autosave("m", "omf", ["Text"], 10e-12)
#autosave("m", "ovf", ["Text"], 10e-12)
#autosave("m", "bin", [], 10e-12)
autotabulate(["t", "H_ext"], "t.txt", 10e-12)
for i in range(100):
	step()

printstats()






