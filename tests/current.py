from mumax2 import *

setgridsize(128, 128, 1)
setcellsize(1e-6, 1e-6, 1e-6)

load('current')
load('solver/euler')

savegraph('graph.png')

setv('dt', 1e-15)


setv('E_ext', [1, 0, 0])

setv('r', 1.7e-2) # Cu
resist = makearray(1, 4, 4, 1)
for i in range(0,4):
	for j in range(0,4):
		resist[0][i][j][0] = 1

resist[0][1][2][0] = 10

setmask('r', resist)
save('E', 'gplot', [])
save('j', 'gplot', [])
save('j.x', 'png', [])
save('j.y', 'png', [])
save('j.z', 'png', [])
save('rho', 'gplot', [])
save('diff_rho', 'gplot', [])


autosave('E.x', 'png', [], 100e-15)
autosave('rho', 'png', [], 100e-15)
autosave('j.x', 'png', [], 100e-15)
autotabulate(['<j>'], 'j.txt', 10e-15)
autosave('diff_rho', 'png', [], 100e-15)

run(100e-12)
