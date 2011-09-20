from mumax2 import *

setgridsize(8, 4, 1)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

#setscalar('alpha', 0.3)
#print 'alpha', getvalue('alpha'), '\n'
#print 'alphaMask', getmask('alpha'), '\n'
#print 'alpha', getfield('alpha'), '\n'

setscalar('Msat', 800e3)
print 'Msat', getvalue('Msat'), '\n'

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setfield('m', m)
print 'm', getfield('m'), '\n'

Hx = 0 / mu0
Hy = 0 / mu0
Hz = 0.1 / mu0 #1T

setvalue('H_z', [Hx, Hy, Hz])
#mask = [ [ [[0]],[[0]] ], [ [[0]], [[0]] ], [ [[1]], [[0]] ] ]
#setmask('H_z', mask)
#print 'H_z',getvalue('H_z'), '\n'
#print 'H', getfield('H'), '\n'
#torque=getfield('torque')
#print 'torque', torque , '\n'
#setfield('torque', m) # must fail

setscalar('dt', 1e-12)
f = open('ll', 'w')
for i in range(1000):
	t = getscalar('t')
	m = probecell('m', 0,0,0)
	f.write(str(t) + "\t")
	f.write(str(m[0]) + "\t")
	f.write(str(m[1]) + "\t")
	f.write(str(m[2]) + "\n")
	step()

f.close()
