from mumax2 import *

setgridsize(8, 4, 1)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

#setscalar('alpha', 0.1)
#print 'alpha', getvalue('alpha'), '\n'
#print 'alphaMask', getmask('alpha'), '\n'
#print 'alpha', getfield('alpha'), '\n'


m=[ [[[1]]], [[[0]]], [[[0]]] ]
setfield('m', m)
print 'm', getfield('m'), '\n'

Hx = 0
Hy = 0
Hz = 1

setvalue('H_z', [Hx, Hy, Hz])
#mask = [ [ [[0]],[[0]] ], [ [[0]], [[0]] ], [ [[1]], [[0]] ] ]
#setmask('H_z', mask)
print 'H_z',getvalue('H_z'), '\n'

#print 'H', getfield('H'), '\n'

torque=getfield('torque')
print 'torque', torque , '\n'

#setfield('torque', m) # must fail

step()
print 'm', getfield('m'), '\n'

