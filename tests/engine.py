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

#step()
#step()
