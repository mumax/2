from mumax2 import *

setgridsize(2, 8, 16)
savegraph("graph.dot")

setscalar('alpha', 0.01)
print 'alpha', getscalar('alpha')

setscalar('msat', 800e3)
print 'msat', getscalar('msat')

setscalar('aexch', 12e-13)
print 'aexch', getscalar('aexch')

m=getfield('m')
m[0][0][0][0] = 1
setfield('m', m)

print 'm', getfield('m')

print 'H', getfield('H')
