from mumax2 import *
from mumax2_geom import *

# test file for demag field

setgridsize(8, 8, 1)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
load('demagexch')

setscalar('Msat', 800e3)
setscalar('Aex', 12e-13)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)

save("kern_ex", "txt", [], "kern_ex.txt")
save("kern_d", "txt", [], "kern_d.txt")
save("kern_dex", "txt", [], "kern_dex.txt")
save("H_dex", "txt", [], "H_dex.txt")

#H = getarray('H')
#print H
echo("")



savegraph("graph.png")
printstats()
