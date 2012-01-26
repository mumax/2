from mumax2 import *

# Test for Faraday's law

Nx = 32 
Ny = 32
Nz = 32
setgridsize(Nx, Ny, Nz)
Cx = 5e-9
Cy = 5e-9
Cz = 5e-9
setcellsize(Cx, Cy, Cz)

load('oersted')
savegraph("graph.png")

wire = makearray(3, Nx, Ny, Nz)
j=Ny/2
k=Nz/2
for i in range(0,Nx):
	wire[0][i][j][k] = 1e12

#setv('j', [1, 0, 0]) 
setarray('j', wire)

save('j', 'omf', ['Text'], 'j.omf')
save('B', 'omf', ['Text'], 'B.omf')
