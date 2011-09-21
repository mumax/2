from mumax2 import *

setgridsize(8, 4, 2)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

setscalar('alpha', 0.3)
#print 'alpha', getvalue('alpha'), '\n'
#print 'alphaMask', getmask('alpha'), '\n'
#print 'alpha', getfield('alpha'), '\n'

setscalar('Msat', 800e3)
print 'Msat', getvalue('Msat'), '\n'

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setfield('m', m)

i=0
j=0
k=1
setcell('m', i,j,k, [0,0,7])
m=getfield('m')
print 'm', m, '\n'
print 'm', i, j, k,  '=', m[0][i][j][k], m[1][i][j][k] , m[2][i][j][k]
#print 'getcell', getcell('m', 2,3,0)


Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 #1T

#setvalue('H_z', [Hx, Hy, Hz])
#mask = [ [ [[0]],[[0]] ], [ [[0]], [[0]] ], [ [[1]], [[0]] ] ]
#setmask('H_z', mask)
#print 'H_z',getvalue('H_z'), '\n'
#print 'H', getfield('H'), '\n'
#torque=getfield('torque')
#print 'torque', torque , '\n'
#setfield('torque', m) # must fail

#setscalar('dt', 1e-12)
#f = open('ll', 'w')
#for i in range(100):
#	t = getscalar('t')
#	#m = getcell('m', 0,0,0)
#	f.write(str(t) + "\t")
#	f.write(str(m[0]) + "\t")
#	f.write(str(m[1]) + "\t")
#	f.write(str(m[2]) + "\n")
#	step()
#
#f.close()





