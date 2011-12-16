from mumax2 import *

# test file for FFT benchmarking

Nx = 1024
Ny = 1024
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(5e-9, 5e-9, 5e-9)

load('micromagnetism')
debug_disableupdate('kern_d')
load('demagexch')
load('solver/euler')

setscalar('Msat', 800e3)
setscalar('Aex', 1.3e-11)
setscalar('alpha', 1)
setscalar('dt', 1e-12)
m=[ [[[1]]], [[[1]]], [[[0]]] ]
setarray('m', m)


savegraph("graph.png")

debug_update("h_dex") # warm-up

resettimer("FW_total")
resettimer("FW_zero")
resettimer("FW_copy")
resettimer("FW_insertBlockZ")
resettimer("FW_pad")
resettimer("FW_Transpose")
resettimer("FW_fftY")
resettimer("FW_fftZ")
#resettimer("FW_fftX")




# force update H demag + exchange a few times
for i in range(0,100):
	debug_invalidate("h_dex")
	debug_update("h_dex")

# this will also print the timing:
printstats()

