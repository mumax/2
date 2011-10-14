from mumax2 import *

setgridsize(8, 4, 2)
#print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
#print 'cellsize', getcellsize()

load('micromagnetism')
savegraph("graph.dot")

setscalar('alpha', 0.1)
getscalar('alpha')

#print 'alpha', getvalue('alpha'), '\n'
#print 'alphaMask', getmask('alpha'), '\n'
#print 'alpha', getfield('alpha'), '\n'

setscalar('Msat', 800e3)
#print 'Msat', getvalue('Msat'), '\n'

m=[ [[[0.01]]], [[[0]]], [[[-1]]] ]
setfield('m', m)
save("m", "txt", [], "m.txt")
save("m", "omf", "text", "mt.omf")
save("m", "omf", "binary 4", "mb.omf")
save("m", "omf", [], "m.omf")

i=3
j=2
k=1
setcell('m', i,j,k, [0,1,0])
#m=getfield('m')
#print 'm', m, '\n'
#print 'm', i, j, k,  '=', m[0][i][j][k], m[1][i][j][k] , m[2][i][j][k]
#print 'getcell', getcell('m', i,j,k)


Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 #1T

setvalue('H_ext', [Hx, Hy, Hz])
#mask = [ [ [[0]],[[0]] ], [ [[0]], [[0]] ], [ [[1]], [[0]] ] ]
#setmask('H_z', mask)
#print 'H_ext',getvalue('H_ext'), '\n'
#print 'H', getfield('H'), '\n'
#torque=getfield('torque')
#print 'torque', torque , '\n'
#setfield('torque', m) # must fail

setscalar('dt', 1e-12)
f = open('ll', 'w')
invalidate('H_ext')
update('H_ext')
invalidate('H_ext')
autosave1=autosave("m", "omf", [], 100e-12)
filenumberformat("%08d")
autosave2=autotabulate(["t", "H_ext"], "t.txt", 10e-12)
for i in range(100):
	step()



printstats()


f.close()






