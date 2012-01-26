from mumax2 import *
from mumax2_geom import *

# Test for Faraday's law

Nx = 32 
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
Cx = 5e-9
Cy = 5e-9
Cz = 5e-9
setcellsize(Cx, Cy, Cz)

load('faraday')
savegraph("graph.png")
setpointwise('B_ext', 0, [0, 0, 0])
setpointwise('B_ext', 1e-9, [0, 0, 1]) 
setpointwise('B_ext', 2e-9, [0, 0, 0]) 
e=ellipse()
disk=makearray(3, 1, 1, 1)
disk[0]=e[0]
disk[1]=e[0]
disk[2]=e[0]
setmask('B_ext', disk)

autotabulate(['t', 'B_ext', '<B>', '<dB_dt>'], 'B.txt', 1e-12)
autosave('B', 'omf', ['Text'], 1e-12)
autosave('B', 'gplot', [], 0.1e-12)
autosave('dB_dt', 'omf', ['Text'], 1e-12)
autosave('dB_dt', 'gplot', [], 0.1e-12)
autosave('E', 'omf', ['Text'], 1e-12)
autosave('E', 'gplot', [], 0.1e-12)
setv('dt', 1e-12)
#run(100e-12)
steps(10)

E=getarray('E')

# test E vector
dBdt=1e9
i = Nx / 4
x = (-Nx/2 + i) * Cx
echo("x=" + str(x))
S = pi * x * x
emf = dBdt * S
Egood = emf / (2 * pi * x)
Ehave = E[1][i][0][0]
echo("Ehave " + str(Ehave) + " Ewant " + str(Egood))


x0 = 3
y0 = 9
x1 = 8
y1 = 14

emf = 0
for i in range(x0,x1):
	j = y0
	emf += Cx*E[0][i][j][0]

i = x1
for j in range(y0,y1):
	emf += Cy*E[1][i][j][0]

j=y1
for i in range(x0,x1):
	j = y0
	emf -= Cx*E[0][i][j][0]

i=x0
for j in range(y0,y1):
	emf -= Cy*E[1][i][j][0]
	

S = (x1-x0-1)*Cx * (y1-y0)*Cy
want = S * dBdt
echo("emf " + str(emf) + " want: " + str(want))
dB = getv("<dB_dt>")
echo("<dBdt> " + str(dB))

