from mumax2 import *

setgridsize(8, 4, 1)
print 'gridsize', getgridsize()

setcellsize(5e-9, 5e-9, 50e-9)
print 'cellsize', getcellsize()

load('test')
savegraph("graph.dot")

m=[ [[[1]]], [[[2]]], [[[3]]] ]
setfield('m', m)
m=getfield('m')
print 'm', m
print


Bx = 1
By = 2
Bz = 3

setvalue('h_z', [Bx, By, Bz])
hz = getvalue('h_z')
print 'h_z', hz
print

h=getfield('h')
print 'h', h
print

torque=getfield('torque')
print 'torque', torque 
print

#step()
#step()
