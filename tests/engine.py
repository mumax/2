from mumax2 import *

setgridsize(8, 2, 1)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

m=[ [ [[111],[111]],  [[111],[111]],  [[111],[111]] ] , 
    [ [[222],[222]],  [[222],[222]],  [[222],[222]] ] , 
    [ [[333],[333]],  [[333],[333]],  [[333],[333]] ] ]

setfield('m', m)

m=getfield('m')
print 'm', m

step()
step()
