from mumax2 import *
from mumax2_geom import *

# test file for demag field

setgridsize(32, 32, 1)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('demagexch')

setscalar('Msat', 800e3)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)

save("H", "txt", [], "H.txt")
H = getarray('H')
print H
echo("")

printstats()
savegraph("graph.png")

