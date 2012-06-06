from mumax2 import *
from sys import exit 
from math import *

# test file for demag field

setgridsize(32, 32, 32)
setcellsize(5e-9, 5e-9, 5e-9)

load('demag')

setscalar('Msat', 800e3)

m=[ [[[1]]], [[[0]]], [[[0]]] ]
setarray('m', m)

hdex = getv('<B>')
hdex[0] /= mu0
hdex[1] /= mu0
hdex[2] /= mu0
echo("H " + str(hdex)) 

tolerance=10

if abs(hdex[1]) > tolerance or abs(hdex[2]) > tolerance:
		exit(-1)

Hx_good = -800e3/3

if abs(hdex[0] - Hx_good) > tolerance:
		exit(-2)

echo("")



savegraph("graph.png")
#printstats()
