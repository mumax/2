from mumax2 import *

setgridsize(8, 4, 2)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

m=[ [[[1]]], [[[0]]], [[[0]]] ]

setfield('m', m)

m=getfield('m')
print 'm', m

Bx = 0
By = 0
Bz = 100e3

setvalue('H_z', [Bx/mu0, By/mu0, Bz/mu0])

#step()
#step()
