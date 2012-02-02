from mumax2 import *

setgridsize(128, 128, 1)
setcellsize(1e-6, 1e-6, 1e-6)

load('current')
savegraph('graph.png')

setcell('rho', 0, 0, 0, [1e-8])

setv('r', 1.7e-8) # Cu
resist = makearray(1, 1, 1, 1)
resist[0][0][0][0] = 1
setmask('r', resist)
save('E', 'gplot', [], 'E.gplot')
save('E', 'omf', ["Text"], 'E.omf')
save('j', 'gplot', [], 'j.gplot')

printstats()
