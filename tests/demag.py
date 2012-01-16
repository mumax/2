from mumax2 import *
from sys import exit 
from math import *

# test file for demag field

# thin film
setgridsize(16, 16, 1)
#setgridsize(16, 4, 1)
setcellsize(5e-9, 5e-9, 2e-9)

load('micromagnetism')
load('demagexch')

setscalar('Msat', 800e3)
setscalar('Aex', 0)

m=[ [[[0]]], [[[0]]], [[[1]]] ]
setarray('m', m)

hdex = getvalue('<h_dex>')
echo("h_dex " + str(hdex)) 

if hdex[0] != 0 or hdex[1] != 0:
		exit(-1)

Hz_good = -790000

if abs(hdex[2] - Hz_good) > 1000:  
		exit(-2)

echo("")



savegraph("graph.png")
#printstats()
