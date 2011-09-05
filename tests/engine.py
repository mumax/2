from mumax2 import *

setgridsize(2, 64, 64)
savegraph("graph.dot")

setscalar('alpha', 0.01)
print 'alpha', getscalar('alpha')

setscalar('msat', 800e3)
print 'msat', getscalar('msat')

setscalar('aexch', 12e-13)
print 'aexch', getscalar('aexch')

print 'm', getfield('m')
