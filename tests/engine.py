from mumax2 import *

setgridsize(8, 4, 2)
#print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
#print 'cellsize', getcellsize()

load('micromagnetism')
savegraph("graph.dot")

setscalar('alpha', 0.05)
getscalar('alpha')

#print 'alpha', getvalue('alpha'), '\n'
#print 'alphaMask', getmask('alpha'), '\n'
#print 'alpha', getfield('alpha'), '\n'

setscalar('Msat', 800e3)
#print 'Msat', getvalue('Msat'), '\n'

m=[ [[[0.01]]], [[[0]]], [[[-1]]] ]
setfield('m', m)
save("m", "ascii", "m.txt")

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

setscalar('dt', 2e-12)
f = open('ll', 'w')
invalidate('H_ext')
update('H_ext')
invalidate('H_ext')
for i in range(30):
	echo("id " + str(outputid()))
	t = getscalar('t')
	m = getcell('m', 0,0,0)
	f.write(str(t) + "\t")
	f.write(str(m[0]) + "\t")
	f.write(str(m[1]) + "\t")
	f.write(str(m[2]) + "\n")
	#invalidate('H_ext')
	#echo('H: ' + str(debugfield('H')))
	step()
	#printstats()



printstats()


f.close()






