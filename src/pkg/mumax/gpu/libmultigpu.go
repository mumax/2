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

// Partial sums (see reduce.h)
func PartialSum(in, out *Array, blocks, threadsPerBlock, N int) {
	C.partialSumAsync(
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial maxima (see reduce.h)
func PartialMax(in, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxAsync(
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial minima (see reduce.h)
func PartialMin(in, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMinAsync(
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial maxima of absolute values (see reduce.h)
func PartialMaxAbs(in, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxAbsAsync(
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial maximum difference between arrays (see reduce.h)
func PartialMaxDiff(a, b, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxDiffAsync(
		(**C.float)(unsafe.Pointer(&a.pointer[0])),
		(**C.float)(unsafe.Pointer(&b.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Copy from src to dst, which have different size3D[Z].
// If dst is smaller, the src input is cropped to the right size.
// If dst is larger, the src input is padded with zeros to the right size.
func CopyPadZ(dst, src *Array) {
	Assert(
		dst.size4D[0] == src.size4D[0] &&
			dst.size3D[0] == src.size3D[0] &&
			dst.size3D[1] == src.size3D[1])

	D2 := dst.size3D[2]
	S0 := src.size4D[0] * src.size3D[0] // NComp * Size0
	S1Part := src.partSize[1]
	S2 := src.size3D[2]
	C.copyPadZAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1Part),
		C.int(S2),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
	dst.Stream.Sync()
}

func TransposeComplexYZPart(out, in *Array) {
//	Assert(
//		out.size4D[0] == in.size4D[0] &&
//			out.size3D[0] == in.size3D[0] &&
//			out.size3D[1] == in.size3D[2]/2 &&
//			out.size3D[2] == in.size3D[1]*2)

	C.transposeComplexYZAsyncPart(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(out.size4D[0]*out.size3D[0]), // nComp * N0
		C.int(in.size3D[1]),
		C.int(in.size3D[2]), // not / 2 !
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}
