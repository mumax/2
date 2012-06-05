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

jx=1e12 #A/m
wire = makearray(3, Nx, Ny, Nz)
j=Ny/2
k=Nz/2
for i in range(0,Nx):
	wire[0][i][j][k] = 1

setv('j', [jx, 0, 0]) 
setmask('j', wire)

save('j', 'omf', ['Text'])
save('B', 'omf', ['Text'])

B=getarray('B')
i=Nx/2
j=Ny/4
k=Nz/2
have=B[2][i][j][k]
y=Cy * (j - Ny/2)
echo("y=" + str(y) + "m")
I=jx*Cy*Cz
echo("I=" + str(I) + " A")
want = (mu0*I)/(2*pi*y)
echo ("want:" + str(want) + "T, have: " + str(have) + "T")
