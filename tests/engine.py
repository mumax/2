from mumax2 import *

setgridsize(16, 8, 2)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

modprobe('micromag')
modprobe('spintorque')
savegraph("graph.dot")

setscalar('alpha', 0.01)
print 'alpha', get('alpha')

setscalar('msat', 800e3)
print 'msat', getscalar('msat')

setscalar('aexch', 12e-13)
print 'aexch', getscalar('aexch')

m=getfield('m')
m[0][0][0][0] = 1
setfield('m', m)

#print 'm', getfield('m')

#print 'H', getfield('H')
