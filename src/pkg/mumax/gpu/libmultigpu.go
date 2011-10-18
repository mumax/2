//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file wraps libmultigpu.so.
// Author: Arne Vansteenkiste

package gpu

//#include "libmultigpu.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

// Adds 2 multi-GPU arrays: dst = a + b
func Add(dst, a, b *Array) {
	C.addAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Multiply-add: dst = a + mulB*b
// b may contain NULL pointers, implemented as all 1's.
func Madd(dst, a, b *Array, mulB float32) {
	C.maddAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Calculates:
//	τ = (m x h) - α m  x (m x h)
// If h = H/Msat, then τ = 1/gamma*dm/dt
func Torque(τ, m, h, αMap *Array, αMul float32) {
	C.torqueAsync(
		(**C.float)(unsafe.Pointer(&(τ.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(τ.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(τ.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(αMap.pointer[0]))),
		(C.float)(αMul),

		(*C.CUstream)(unsafe.Pointer(&(τ.Stream[0]))),

		(C.int)(m.partLen3D))
	τ.Stream.Sync()

}

// Normalize
func Normalize(m, normMap *Array) {
	C.normalizeAsync(
		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(normMap.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(m.Stream[0]))),
		C.int(m.partLen4D))
	m.Stream.Sync()
}

func PartialSum(in, out *Array, blocks, threadsPerBlock, N int) {
		C.partialSum(
			(**C.float)(unsafe.Pointer(&in.pointer[0])),
			(**C.float)(unsafe.Pointer(&out.pointer[0])),
			C.int(blocks),
			C.int(threadsPerBlock),
			C.int(N),
			(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
		out.Stream.Sync()
}
