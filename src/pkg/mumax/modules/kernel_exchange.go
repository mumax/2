//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Kernel for exchange interaction
// Author: Arne Vansteenkiste, Ben Van de Wiele

import (
	. "mumax/common"
	"mumax/host"
	"math"
)


// Many-neighbor exchange kernel
// range2: square of range in which neighbors are considered.
// TODO: check against donahue!
// need extra penalty for number of cells > 4 ?
func ExchKernel(size []int, cellsize []float64, kern *host.Array, range2 float64) {
	Debug("Calculating exchange kernel", "range²:", range2)
	Start("kern_ex")

	if range2 < 1 {
		panic(InputErrF("Exchange range should be at least 1"))
	}
	// N = range2 should be int
	N := int(range2)
	Assert(float64(N) == range2)
	R := int(math.Sqrt(range2) + 1) // upper bound for range in number of cells


	var totalWeight float64
	for i := 1; i <= N; i++ {
		totalWeight += 1 / float64(i)
	}
	scale := 1 / totalWeight

	xmin, xmax := -R, R
	ymin, ymax := -R, R
	zmin, zmax := -R, R
	if size[X] == 1 { // 2D case
		xmin, xmax = 0, 0
	}
	Nneigh := 0 // counts number of neighbors, debug

	dx := cellsize[X]
	dy := cellsize[Y]
	dz := cellsize[Z]

	for s := 0; s < 3; s++ { // source index Ksdxyz
		i := TensorIdx[s][s]
		arr := kern.Array[i]
		var total float64

		for i := xmin; i <= xmax; i++ {
			for j := ymin; j <= ymax; j++ {
				for k := zmin; k <= zmax; k++ {
					if i*i+j*j+k*k > N {
						continue //only look at close enough neighbors
					}
					if !(i == 0 && j == 0 && k == 0) {
						n := i*i + j*j + k*k // distance² from center, in #cells
						lapl := 1 / (sqr(float64(i)*dx) + sqr(float64(j)*dy) + sqr(float64(k)*dz))
						val := lapl * (1 / float64(n)) * scale
						total += val
						arr[Wrap(i, size[X])][Wrap(j, size[Y])][Wrap(k, size[Z])] = float32(val)
						if s == 0 { // count neighbors for one component only
							Nneigh++
						}
					}
				}
			}
		}
		arr[0][0][0] = float32(-total)
	}
	Stop("kern_ex")
	//if N != 1{
	Log("Exchange #neighbors:", Nneigh)
	//}
}

func sqr(x float64) float64 { return x * x }


// 6-Neighbor Laplace kernel used for exchange.
//
// Note on self-contributions and the energy density:
//
// Contributions to H_eff that are parallel to m do not matter.
// They do not influence the dynamics and only add a constant term to the energy.
// Therefore, the self-contribution of the exchange field can be neglected. This
// term is -N*m for a cell in a cubic grid, with N the number of neighbors.
// By neglecting this term, we do not need to take into account boundary conditions.
// Because the interaction can then be written as a convolution, we can simply
// include it in the demag convolution kernel and we do not need a separate calculation
// of the exchange field anymore: an elegant and efficient solution.
// The dynamics are still correct, only the total energy is offset with a constant
// term compared to the usual - M . H. Outputting H_eff becomes less useful however,
// it's better to look at torques. Away from the boundaries, H_eff is as usual.
func Exch6NgbrKernel(size []int, cellsize []float64, kern *host.Array) {
	Debug("Calculating laplace 6 kernel", "size:", size, "cellsize:", cellsize)
	Start("kern_ex")

	for s := 0; s < 3; s++ { // source index Ksdxyz
		i := TensorIdx[s][s]
		arr := kern.Array[i]

		hx := cellsize[X] * cellsize[X]
		hy := cellsize[Y] * cellsize[Y]
		hz := cellsize[Z] * cellsize[Z]

		arr[Wrap(0, size[X])][Wrap(0, size[Y])][Wrap(0, size[Z])] = float32(-2/hx - 2/hy - 2/hz)
		arr[Wrap(+1, size[X])][Wrap(0, size[Y])][Wrap(0, size[Z])] = float32(1 / hx)
		arr[Wrap(-1, size[X])][Wrap(0, size[Y])][Wrap(0, size[Z])] = float32(1 / hx)
		arr[Wrap(0, size[X])][Wrap(+1, size[Y])][Wrap(0, size[Z])] = float32(1 / hy)
		arr[Wrap(0, size[X])][Wrap(-1, size[Y])][Wrap(0, size[Z])] = float32(1 / hy)
		arr[Wrap(0, size[X])][Wrap(0, size[Y])][Wrap(+1, size[Z])] = float32(1 / hz)
		arr[Wrap(0, size[X])][Wrap(0, size[Y])][Wrap(-1, size[Z])] = float32(1 / hz)
	}
	Stop("kern_ex")
}
