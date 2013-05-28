//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for temperature.cu
// Author: Arne Vansteenkiste

//#include "libmumax2.h"
import "C"
import (
	. "mumax/common"
	"unsafe"
)

func ScaleNoise(noise, alphaMask *Array,
	tempMask *Array, alphaKB2tempMul float32,
	mSatMask *Array, mu0VgammaDtMsatMul float32) {
	CheckSize(noise.Size4D(), alphaMask.Size4D())
	C.temperature_scaleNoise(
		(**C.float)(unsafe.Pointer(&(noise.Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(alphaMask.Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(tempMask.Pointers()[0]))),
		(C.float)(alphaKB2tempMul),
		(**C.float)(unsafe.Pointer(&(mSatMask.Pointers()[0]))),
		(C.float)(mu0VgammaDtMsatMul),
		(*C.CUstream)(unsafe.Pointer(&(noise.Stream[0]))),
		(C.int)(noise.PartLen3D()))
}

func ScaleNoiseAniz(h, mu, T, msat, msat0T0 *Array,
	muMul []float64,
	KB2tempMul float64, mu0VgammaDtMsatMul float64) {
	CheckSize(h.Size3D(), mu.Size3D())
	C.temperature_scaleAnizNoise(
		(**C.float)(unsafe.Pointer(&(h.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(mu.Comp[XX].Pointers()[0]))), //XX
		(**C.float)(unsafe.Pointer(&(mu.Comp[YY].Pointers()[0]))), //YY
		(**C.float)(unsafe.Pointer(&(mu.Comp[ZZ].Pointers()[0]))), //ZZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[YZ].Pointers()[0]))), //YZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[XZ].Pointers()[0]))), //XZ
		(**C.float)(unsafe.Pointer(&(mu.Comp[XY].Pointers()[0]))), //XY

		(**C.float)(unsafe.Pointer(&(T.Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(msat.Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(msat0T0.Pointers()[0]))),

		(C.float)(float32(muMul[XX])),
		(C.float)(float32(muMul[YY])),
		(C.float)(float32(muMul[ZZ])),
		(C.float)(float32(muMul[YZ])),
		(C.float)(float32(muMul[XZ])),
		(C.float)(float32(muMul[XY])),

		(C.float)(float32(KB2tempMul)),
		(C.float)(float32(mu0VgammaDtMsatMul)),
		(*C.CUstream)(unsafe.Pointer(&(h.Stream[0]))),
		(C.int)(h.PartLen3D()))
}
