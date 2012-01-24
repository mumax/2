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
)

// Calculates a biot-savart type kernel
//	-(r x j)/r³ dV
// j is the source (current or displacement current)
// To be used for both Oersted fields and Faraday's law.
//
// TODO: make more accurate, current implementation is that of point source.
func RotorKernel(size []int, cellsize []float64, periodic []int, accuacy int, kern *host.Array) {
	Debug("Calculating rotor kernel", "size:", size, "cellsize:", cellsize, "periodic:", periodic)
	Start("kern_rot")
	k := kern.Array

	Assert(len(kern.Array) == 9)
	CheckSize(kern.Size3D, size)

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
	V := cellsize[X] * cellsize[Y] * cellsize[Z]

	for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped. It's crucial that the unused rows remain zero, otherwise the FFT'ed kernel is not purely real anymore.
		xw := Wrap(x, size[X])
		for y := y1; y <= y2; y++ {
			yw := Wrap(y, size[Y])
			for z := z1; z <= z2; z++ {
				zw := Wrap(z, size[Z])

				var r vector
				r[X], r[Y], r[Z] = float64(x)*cellsize[X], float64(y)*cellsize[Y], float64(z)*cellsize[Z]
				norm := r.Norm()
				r3 := norm * norm * norm

				for s := 0; s < 3; s++ { // source orientations

					// source: unit vector along s direction
					var j vector
					j[s] = 1

					rxj := r.Cross(j)
					(&rxj).Scale(V / r3)

					if norm != 0 {
						for d := 0; d < 3; d++ { // destination orientation
							k[FullTensorIdx[s][d]][xw][yw][zw] += float32(rxj[d])
							// We have to ADD because there are multiple contributions in case of periodicity
						}
					} else {
						// E += 0
					}

				}
			}
		}
	}
	Stop("kern_rot")
}
