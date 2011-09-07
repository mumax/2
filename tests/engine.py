from mumax2 import *

setgridsize(16, 8, 2)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

m=[ [[[ 111 ]]] , [[[ 222 ]]] , [[[ 333 ]]] ]
setfield('m', m)

m=getfield('m')
print 'm', m

step()
step()
