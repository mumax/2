from mumax2 import *

setgridsize(8, 4, 1)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setfield('m', m)


Bx = 0
By = 0
Bz = 1000e-3 

setvalue('h_z', [Bx/mu0, By/mu0, Bz/mu0])

torque=getfield('torque')

print 'torque', torque


#step()
#step()
