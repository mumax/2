from mumax2 import *
from sys import exit 
from math import *

# test file for maxwell demag field

# thin film
setgridsize(128, 128, 1)
#setgridsize(20, 20, 1)
setcellsize(5e-9, 5e-9, 2e-9)

load('demag')

#save('kern_dipole', 'gplot', [], 'kern_dipole.gplot')
setv('Msat', 800e3)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)
setv('B_ext', [2,3,4])

hdex = getvalue('<B>')
echo("B" + str(hdex)) 
#savegraph("graph.png")

if hdex[0] != 2 or hdex[1] != 3:
		exit(-1)

Hz_good = -790000*mu0 + 4

tolerance=1./100
if abs((hdex[2] - Hz_good)/Hz_good) > tolerance:  
		exit(-2)

echo("")



#printstats()
