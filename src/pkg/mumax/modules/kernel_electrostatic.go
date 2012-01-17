//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Kernel for electrical field
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"math"
)

// Calculates the electrostatic kernel: 1/4pi * cell volume / r2
//
// size: size of the kernel, usually 2 x larger than the size of the magnetization due to zero padding
//
// return value: 3 arrays: K[destdir][x][y][z]
// (e.g. K[X][1][2][3] gives H_x at position (1, 2, 3) due to a unit charge density at the origin.
// TODO: make more accurate
func PointKernel(size []int, cellsize []float64, periodic []int, kern *host.Array) {
	Debug("Calculating electrostatic kernel", "size:", size, "cellsize:", cellsize, "periodic:", periodic)
	Start("kern_el")
	k := kern.Array

	Assert(len(kern.Array) == 3)
	CheckSize(kern.Size3D, size)

	//	Warn("OVERRIDING E KERN")
	//	k[X][0][0][0] = 1
	//	k[Y][0][0][0] = 1
	//	k[Z][0][0][0] = 1
	//	return

	// Kernel size: calc. between x1,x2; y1,y2; z1,z2
	x1 := -(size[X] - 1) / 2
	x2 := size[X]/2 - 1
	// support for 2D simulations (thickness 1)
	if size[X] == 1 && periodic[X] == 0 {
		x2 = 0
	}

	y1 := -(size[Y] - 1) / 2
	y2 := size[Y]/2 - 1

	z1 := -(size[Z] - 1) / 2
	z2 := size[Z]/2 - 1

	x1 *= (periodic[X] + 1)
	x2 *= (periodic[X] + 1)
	y1 *= (periodic[Y] + 1)
	y2 *= (periodic[Y] + 1)
	z1 *= (periodic[Z] + 1)
	z2 *= (periodic[Z] + 1)
	Debug("xyz ranges:", x1, x2, y1, y2, z1, z2)

	// cell volume
	//V := cellsize[X] * cellsize[Y] * cellsize[Z]

	for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
		xw := Wrap(x, size[X])
		for y := y1; y <= y2; y++ {
			yw := Wrap(y, size[Y])
			for z := z1; z <= z2; z++ {
				zw := Wrap(z, size[Z])

				rx, ry, rz := float64(x)*cellsize[X], float64(y)*cellsize[Y], float64(z)*cellsize[Z]
				r := math.Sqrt(rx*rx + ry*ry + rz*rz)
				if r != 0 {
					//factor := V / (4 * PI) // TODO: include epsillon0?
					Ex := rx //factor * rx / (r * r * r)
					Ey := ry //factor * ry / (r * r * r)
					Ez := rz //factor * rz / (r * r * r)

					k[X][xw][yw][zw] += float32(Ex)
					k[Y][xw][yw][zw] += float32(Ey)
					k[Z][xw][yw][zw] += float32(Ez)
					// We have to ADD because there are multiple contributions in case of periodicity
				} else {
					// E += 0
				}
			}
		}
	}
	Stop("kern_el")
}
