from mumax2 import *
from sys import exit 
from math import *

# test file for demag field

setgridsize(32, 32, 32)
#setgridsize(20, 20, 2)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')

setscalar('Msat', 800e3)
setscalar('Aex', 0)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

hdex = getvalue('<h_dex>')
echo("h_dex " + str(hdex)) 

tolerance=10

if abs(hdex[1]) > tolerance or abs(hdex[2]) > tolerance:
		exit(-1)

Hx_good = -800e3/3

if abs(hdex[0] - Hx_good) > tolerance:
		exit(-2)

echo("")



savegraph("graph.png")
#printstats()
