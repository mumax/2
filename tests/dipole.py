from mumax2 import *
from sys import exit 
from math import *

# test file for maxwell demag field

# thin film
setgridsize(128, 128, 1)
#setgridsize(20, 20, 1)
setcellsize(5e-9, 5e-9, 2e-9)

load('demag')

setscalar('Msat', 800e3)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)

hdex = getvalue('<B>')
echo("B" + str(hdex)) 
savegraph("graph.png")

if hdex[0] != 0 or hdex[1] != 0:
		exit(-1)

Hz_good = -790000

if abs(hdex[2] - Hz_good) > 1000:  
		exit(-2)

echo("")



#printstats()
