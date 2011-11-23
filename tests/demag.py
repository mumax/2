from mumax2 import *
from mumax2_geom import *

# test file for demag field

setgridsize(32, 32, 1)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('demagexch')

setscalar('Msat', 800e3)
setscalar('Aex', 12e-13)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)


debug_update("kern_d")
debug_update("kern_d.xx")
save("kern_d.xx", "txt", [], "kern_d.xx.txt")

#H = getarray('H')
#print H
echo("")

printstats()
savegraph("graph.png")

