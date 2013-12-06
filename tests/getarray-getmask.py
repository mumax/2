# -*- coding: utf-8 -*-

from mumax2 import *
from random import *
from math import *

eps = 1e-7
eps1 = 0.1
dirty = 0

# material
thickness = 15e-9

setgridsize(32, 16, 64)
setcellsize(2e-9, 3e-9, 4e-9)

load('exchange6')

m = [[[[1.0]]],[[[0.0]]],[[[0.0]]]]
setarray('m', m)

Ms = 8e5

msat = [[[[1.0]]]]
setmask('msat', msat)
setv('msat', Ms)

msMsk = getmask('msat')
msArr = getarray('msat')

for i in range(32):
    for j in range(16):
        for k in range(64):
            val = abs(Ms * msMsk[0][i][j][k] - msArr[0][i][j][k])
            if val > eps:
                dirty = dirty + 1

for i in range(32):
    for j in range(16):
        for k in range(64):
            msMsk[0][i][j][k] = random()
setmask('msat', msMsk)

msMsk1 = getmask('msat')
msArr1 = getarray('msat')

for i in range(32):
    for j in range(16):
        for k in range(64):
            val = abs(Ms * msMsk1[0][i][j][k] - msArr1[0][i][j][k])
            if val > eps1:
                print val
                dirty = dirty + 1

if dirty > 0:
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    sys.exit(1)
else:
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()

