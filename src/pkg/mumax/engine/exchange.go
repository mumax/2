//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	"mumax/host"
)

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
		i := kernIdx[s][s]
		arr := kern.Array[i]

		hx := cellsize[X] * cellsize[X]
		hy := cellsize[Y] * cellsize[Y]
		hz := cellsize[Z] * cellsize[Z]

		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = float32(-2./hx - 2./hy - 2./hz)
		arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = float32(1. / hx)
		arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = float32(1. / hx)
		arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = float32(1. / hy)
		arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = float32(1. / hy)
		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = float32(1. / hz)
		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = float32(1. / hz)
	}
	Stop("kern_ex")
}

// See Donahue, M. J. & Porter, D. G.
// Exchange energy formulations for 3D micromagnetics
// Physica B-condensed Matter, 2004, 343, 177-183
//func Exch12NgbrKernel(size []int, cellsize []float32) []*tensor.T3 {
//	k := make([]*tensor.T3, 6)
//	for i := range k {
//		k[i] = tensor.NewT3(size)
//	}
//
//	hx := cellsize[X] * cellsize[X]
//	hy := cellsize[Y] * cellsize[Y]
//	hz := cellsize[Z] * cellsize[Z]
//
//	for s := 0; s < 3; s++ { // source index Ksdxyz
//		i := KernIdx[s][s]
//		arr := k[i].Array()
//
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = -2.5/hx - 2.5/hy - 2.5/hz
//
//		arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 4. / 3. / hx
//		arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 4. / 3. / hx
//		arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 4. / 3. / hy
//		arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 4. / 3. / hy
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 4. / 3. / hz
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 4. / 3. / hz
//
//		arr[wrap(2, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = -1. / 12. / hx
//		arr[wrap(-2, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = -1. / 12. / hx
//		arr[wrap(0, size[X])][wrap(2, size[Y])][wrap(0, size[Z])] = -1. / 12. / hy
//		arr[wrap(0, size[X])][wrap(-2, size[Y])][wrap(0, size[Z])] = -1. / 12. / hy
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(2, size[Z])] = -1. / 12. / hz
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(-2, size[Z])] = -1. / 12. / hz
//	}
//
//	return k
//}
//
//// See Donahue, M. J. & Porter, D. G.
//// Exchange energy formulations for 3D micromagnetics
//// Physica B-condensed Matter, 2004, 343, 177-183
//func Exch26NgbrKernel(size []int, cellsize []float32) []*tensor.T3 {
//	k := make([]*tensor.T3, 6)
//	for i := range k {
//		k[i] = tensor.NewT3(size)
//	}
//
//	hx := 18. * cellsize[X] * cellsize[X]
//	hy := 18. * cellsize[Y] * cellsize[Y]
//	hz := 18. * cellsize[Z] * cellsize[Z]
//
//	for s := 0; s < 3; s++ { // source index Ksdxyz
//		i := KernIdx[s][s]
//		arr := k[i].Array()
//
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = -32./hx - 32./hy - 32./hz
//
//		arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 16./hx - 8./hy - 8./hz
//		arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 16./hx - 8./hy - 8./hz
//		arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = -8./hx + 16./hy - 8./hz
//		arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = -8./hx + 16./hy - 8./hz
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = -8./hx - 8./hy + 16./hz
//		arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = -8./hx - 8./hy + 16./hz
//
//		arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 4./hx + 4./hy - 2./hz
//		arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 4./hx + 4./hy - 2./hz
//		arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 4./hx + 4./hy - 2./hz
//		arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 4./hx + 4./hy - 2./hz
//
//		arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 4./hx - 2./hy + 4./hz
//		arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 4./hx - 2./hy + 4./hz
//		arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 4./hx - 2./hy + 4./hz
//		arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 4./hx - 2./hy + 4./hz
//
//		arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = -2./hx + 4./hy + 4./hz
//		arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = -2./hx + 4./hy + 4./hz
//		arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = -2./hx + 4./hy + 4./hz
//		arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = -2./hx + 4./hy + 4./hz
//
//		arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = 1./hx + 1./hy + 1./hz
//		arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = 1./hx + 1./hy + 1./hz
//
//	}
//
//	//   for s := 0; s < 3; s++ { // source index Ksdxyz
//	//     i := KernIdx[s][s]
//	//     arr := k[i].Array()
//	// 
//	//     arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] =1.
//	// 
//	//     arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 1.
//	// 
//	//     arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(0, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(0, size[Z])] = 1.
//	// 
//	//     arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(1, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(0, size[Y])][wrap(-1, size[Z])] = 1.
//	// 
//	//     arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(0, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = 1.
//	// 
//	//     arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(1, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = 1.
//	//     arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(1, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(1, size[Y])][wrap(-1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(1, size[Z])] = 1.
//	//     arr[wrap(-1, size[X])][wrap(-1, size[Y])][wrap(-1, size[Z])] = 1.
//	// 
//	//   }
//
//	return k
//}
