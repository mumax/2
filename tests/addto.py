from mumax2 import *

# Tests the add_to() api

Nx = 1024
Ny = 1024
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('zeeman')

add_to('H_eff', 'H_ext')

H=[[[[0.0]]],[[[0.0]]],[[[0.0]]]]
He=[[[[1.0]]],[[[0.5]]],[[[0.1]]]]
setarray('H_eff', H)
setmask('H_ext', He)
setv('H_ext', [1.0, 1.0, 1.0])

Hval = getv('<H_eff>')
print Hval
if Hval[0] != 1.0 or Hval[1] != 0.5 or Hval[2] != 0.1:
    print "error in add_to"
    
add_to_weighted('H_eff', 'H_ext1', -2.0)
setmask('H_ext1', He)
setmask('H_ext', He)
setv('H_ext1', [1.0, 1.0, 1.0]) 
Hval = getv('<H_eff>')
print Hval
if Hval[0] != -1.0 or Hval[1] != -0.5 or Hval[2] != -0.1:
    print "error in add_to_weighted"



