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


print 'm', getfield('m')
print
m=[ [[[1]]], [[[0]]], [[[0]]] ]
setfield('m', m)
print 'm', getfield('m')
print


Bx = 0
By = 0
Bz = 1

setvalue('H_z', [Bx, By, Bz])
mask = [ [ [[0]],[[0]] ], [ [[0]], [[0]] ], [ [[1]], [[0]] ] ]
setmask('H_z', mask)
print 'H_z',getvalue('H_z')
print

print 'H', getfield('H')
print

torque=getfield('torque')
print 'torque', torque 
print

#setfield('torque', m) # must fail

#step()
#step()
