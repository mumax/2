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


// add uniaxial anisotropy to h, in units [mSat]
// func AddHaniUniaxial(h, m, kuMap *Array, kuMul float32, anisUMap *Array, anisUMul []float64) {
//   C.addHaniUniaxialAsync(
//     (**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),
// 
//     (**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
// 
//     (**C.float)(unsafe.Pointer(&(kuMap.pointer[0]))),
//     (C.float)(kuMul),
// 
//     (**C.float)(unsafe.Pointer(&(anisUMap.Comp[X].pointer[0]))),
//     (C.float)(anisUMul[0]),
//     (**C.float)(unsafe.Pointer(&(anisUMap.Comp[Y].pointer[0]))),
//     (C.float)(anisUMul[1]),
//     (**C.float)(unsafe.Pointer(&(anisUMap.Comp[Z].pointer[0]))),
//     (C.float)(anisUMul[2]),
// 
//     (*C.CUstream)(unsafe.Pointer(&(h.Stream[0]))),
// 
//     (C.int)(m.partLen3D))
//   h.Stream.Sync()
// 
// }

// add cubic anisotropy to h, in units [mSat]
// func AddHaniCubic(h, m,
//     k1Map *Array, k1Mul float32,
//     k2Map *Array, k2Mul float32,
// anisU1Map *Array, anisU1Mul []float64,
// anisU2Map *Array, anisU2Mul []float64) {
//   C.addHaniCubicAsync(
//     (**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),
// 
//     (**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
//     (**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
// 
//     (**C.float)(unsafe.Pointer(&(k1Map.pointer[0]))),
//     (C.float)(k1Mul),
//     (**C.float)(unsafe.Pointer(&(k2Map.pointer[0]))),
//     (C.float)(k2Mul),
// 
//     (**C.float)(unsafe.Pointer(&(anisU1Map.Comp[X].pointer[0]))),
//     (C.float)(anisU1Mul[0]),
//     (**C.float)(unsafe.Pointer(&(anisU1Map.Comp[Y].pointer[0]))),
//     (C.float)(anisU1Mul[1]),
//     (**C.float)(unsafe.Pointer(&(anisU1Map.Comp[Z].pointer[0]))),
//     (C.float)(anisU1Mul[2]),
//     (**C.float)(unsafe.Pointer(&(anisU2Map.Comp[X].pointer[0]))),
//     (C.float)(anisU2Mul[0]),
//     (**C.float)(unsafe.Pointer(&(anisU2Map.Comp[Y].pointer[0]))),
//     (C.float)(anisU2Mul[1]),
//     (**C.float)(unsafe.Pointer(&(anisU2Map.Comp[Z].pointer[0]))),
//     (C.float)(anisU2Mul[2]),
// 
//     (*C.CUstream)(unsafe.Pointer(&(h.Stream[0]))),
// 
//     (C.int)(m.partLen3D))
//   h.Stream.Sync()
// 
// }


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

// Copy from src into a block in dst
// E.g.:
// 	2x2 source, block = 1, 2x6 dst:
//	[ 0 0  S1 S2  0 0 ]
//	[ 0 0  S3 S4  0 0 ]
func CopyBlockZ(dst, src *Array, block int) {
	Assert(
		dst.size4D[0] == src.size4D[0] &&
			dst.size3D[0] == src.size3D[0] &&
			dst.size3D[1] == src.size3D[1] &&
			dst.size3D[2] >= src.size3D[2]*(block+1))

	D2 := dst.size3D[2]
	S0 := src.size4D[0] * src.size3D[0] // NComp * Size0
	S1Part := src.partSize[1]
	S2 := src.size3D[2]
	C.copyBlockZAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1Part),
		C.int(S2),
		C.int(block),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
	dst.Stream.Sync()
}

//// Transpose parts on each GPU individually.
//// BUG: does not do anything with > 1 GPU.
func TransposeComplexYZPart(out, in *Array) {
	Assert(
		out.size4D[0] == in.size4D[0] &&
			out.size3D[0] == in.size3D[0] &&
			out.size3D[1]*out.size3D[2] == in.size3D[2]*in.size3D[1])

	//Debug("in.partSize[1]", in.partSize[1])
	C.transposeComplexYZAsyncPart(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.partSize[1]),            //!?
		C.int(in.size3D[2]),              // not / 2 !
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

//// Cross-device YZ transpose + pad.
//func TransposeComplexYZ(dst, src *Array) {
//	N0 := src.size4D[0] * src.size3D[0]
//	N1Part := src.size3D[1] / NDevice()
//	N2 := src.size3D[2]
//
//	C.transposePadYZAsync(
//		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
//		(**C.float)(unsafe.Pointer(&src.pointer[0])),
//		C.int(N0),
//		C.int(N1Part),
//		C.int(N2),
//		C.int(N2), //N2Pad
//		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
//	dst.Stream.Sync()
//}

//func CombineZ(dst, src1, src2 *Array) {
//	AssertEqual(src1.size4D, src2.size4D)
//	Assert(dst.size4D[0] == src1.size4D[0] &&
//		dst.size3D[0] == src1.size3D[0] &&
//		dst.size3D[1] == src1.size3D[1] &&
//		dst.size3D[2] == src1.size3D[2]*2)
//
//	D2 := dst.size3D[2]
//	S0 := src1.size4D[0] * src1.size3D[0] // NComp * Size0
//	S1Part := src1.partSize[1]
//	S2 := src1.size3D[2]
//	C.combineZAsync(
//		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
//		C.int(D2),
//		(**C.float)(unsafe.Pointer(&src1.pointer[0])),
//		(**C.float)(unsafe.Pointer(&src2.pointer[0])),
//		C.int(S0),
//		C.int(S1Part),
//		C.int(S2),
//		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
//	dst.Stream.Sync()
//}
