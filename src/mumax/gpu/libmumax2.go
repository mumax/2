//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file wraps the core functions of libmultigpu.so.
// Functions added by add-on modules are wrapped elsewhere.
// Author: Arne Vansteenkiste, Ben Van de Wiele

package gpu

//#cgo LDFLAGS:-Wl,-rpath=\$ORIGIN/../bin/ -L. -lmumax2 -L/usr/local/cuda/lib64 -LC:/opt/cuda/lib/x64 -LC:/opt/cuda/lib/x64 -lcudart
//#cgo CFLAGS:-IC:/opt/cuda/include -I../../libmumax2 -I/usr/local/cuda/include
//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	"unsafe"
)

// Adds 2 multi-GPU arrays: dst = a + b
func Add(dst, a, b *Array) {
	CheckSize(dst.size4D, a.size4D)
	C.addAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Multiply 2 multi-GPU arrays: dst = a * b
func Mul(dst, a, b *Array) {
	CheckSize(dst.size4D, a.size4D)
	C.mulAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Divide 2 multi-GPU arrays: dst = a / b; _if_ b = 0 _then_ dst = 0
func Div(dst, a, b *Array) {
	CheckSize(dst.size4D, a.size4D)
	C.divAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Divide and Multiply by the array raised to the Power : dst = pow(c, p) * a / b; _if_ b = 0 _then_ dst = a, _if_c = 0 _then_ dst = 0
func DivMulPow(dst, a, b, c *Array, p float64) {
	CheckSize(dst.size4D, a.size4D)
	C.divMulPowAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(c.pointer[0]))),
		C.float(float32(p)),

		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Synchronous Dot product: C = AiBi, A and B could be masks
func DotMask(dst, a, b *Array, aMul, bMul []float64) {
	CheckSize(dst.size3D, a.size3D)
	C.dotMaskAsync(
		(**C.float)(unsafe.Pointer(&(dst.Comp[X].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(a.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(b.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Z].Pointers()[0]))),

		(C.float)(float32(aMul[X])),
		(C.float)(float32(aMul[Y])),
		(C.float)(float32(aMul[Z])),

		(C.float)(float32(bMul[X])),
		(C.float)(float32(bMul[Y])),
		(C.float)(float32(bMul[Z])),

		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen3D))
	dst.Stream.Sync()
}

// Synchronous Dot product: C = AiBi, A and B should not be masks
func Dot(dst, a, b *Array) {
	CheckSize(dst.size3D, a.size3D)
	C.dotAsync(
		(**C.float)(unsafe.Pointer(&(dst.Comp[X].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(a.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(b.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Z].Pointers()[0]))),

		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen3D))
	dst.Stream.Sync()
}

// Synchronous Singed Dot product: C = sign(BC) * (AB)
func DotSign(dst, a, b, c *Array) {
	CheckSize(dst.size3D, a.size3D)
	C.dotSignAsync(
		(**C.float)(unsafe.Pointer(&(dst.Comp[X].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(a.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(b.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(c.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(c.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(c.Comp[Z].Pointers()[0]))),

		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen3D))
	dst.Stream.Sync()
}

// Asynchronous multiply-add: a += mulB*b
// b may contain NULL pointers, implemented as all 1's.
func MAdd1Async(a, b *Array, mulB float32, stream Stream) {
	C.madd1Async(
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		C.int(a.partLen4D))
}

// Asynchronous multiply-add: a += mulB*b + mulC*c
// b,c may contain NULL pointers, implemented as all 1's.
func MAdd2Async(a, b *Array, mulB float32, c *Array, mulC float32, stream Stream) {
	C.madd2Async(
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(**C.float)(unsafe.Pointer(&(c.pointer[0]))),
		(C.float)(mulC),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		C.int(a.partLen4D))
}

// 3-vector multiply-add: dst_i = a_i + mulB_i*b_i
// b may contain NULL pointers, implemented as all 1's.
func VecMadd(dst, a, b *Array, mulB []float64) {
	C.vecMaddAsync(
		(**C.float)(unsafe.Pointer(&(dst.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(dst.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(dst.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(a.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(a.Comp[Z].Pointers()[0]))),

		(**C.float)(unsafe.Pointer(&(b.Comp[X].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Y].Pointers()[0]))),
		(**C.float)(unsafe.Pointer(&(b.Comp[Z].Pointers()[0]))),

		(C.float)(float32(mulB[X])),
		(C.float)(float32(mulB[Y])),
		(C.float)(float32(mulB[Z])),

		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen3D))
	dst.Stream.Sync()
}

func Madd(dst, a, b *Array, mulB float64) {
	C.maddAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(float32(mulB)),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Multiply-add: dst = a + mul* (b + c)
// b may NOT contain NULL pointers!
func AddMadd(dst, a, b, c *Array, mul float32) {
	C.addMaddAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(c.pointer[0]))),
		(C.float)(mul),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Complex multiply add.
// dst and src contain complex numbers (interleaved format)
// kern contains real numbers
// 	dst[i] += scale * kern[i] * src[i]
func CMaddAsync(dst *Array, scale complex64, kern, src *Array, stream Stream) {
	//	Debug("CMadd dst", dst.Size4D())
	//	Debug("CMadd src", src.Size4D())
	//	Debug("CMadd dst.Len", dst.Len())
	//	Debug("CMadd src.Len", src.Len())
	CheckSize(dst.Size3D(), src.Size3D())
	AssertMsg(dst.Len() == src.Len(), "src-dst")
	AssertMsg(dst.Len() == 2*kern.Len(), "dst-kern")
	C.cmaddAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(C.float)(real(scale)),
		(C.float)(imag(scale)),
		(**C.float)(unsafe.Pointer(&(kern.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(src.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		(C.int)(kern.PartLen3D())) // # of numbers (real or complex)
}

// dst[i] = a[i]*mulA + b[i]*mulB
func LinearCombination2Async(dst *Array, a *Array, mulA float32, b *Array, mulB float32, stream Stream) {
	dstlen := dst.Len()
	Assert(dstlen == a.Len() && dstlen == b.Len())
	C.linearCombination2Async(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(C.float)(mulA),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		C.int(dst.partLen4D))
}

// dst[i] = a[i]*mulA + b[i]*mulB + c[i]*mulC
func LinearCombination3Async(dst *Array, a *Array, mulA float32, b *Array, mulB float32, c *Array, mulC float32, stream Stream) {
	dstlen := dst.Len()
	Assert(dstlen == a.Len() && dstlen == b.Len() && dstlen == c.Len())
	C.linearCombination3Async(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(C.float)(mulA),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(**C.float)(unsafe.Pointer(&(c.pointer[0]))),
		(C.float)(mulC),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		C.int(dst.partLen4D))
}

// dst[i] = a[i]*mulA + b[i]*mulB + c[i]*mulC
func LinearCombination3(dst *Array, a *Array, mulA float32, b *Array, mulB float32, c *Array, mulC float32) {
	dstlen := dst.Len()
	Assert(dstlen == a.Len() && dstlen == b.Len() && dstlen == c.Len())
	C.linearCombination3Async(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(a.pointer[0]))),
		(C.float)(mulA),
		(**C.float)(unsafe.Pointer(&(b.pointer[0]))),
		(C.float)(mulB),
		(**C.float)(unsafe.Pointer(&(c.pointer[0]))),
		(C.float)(mulC),
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

func WeightedAverage(dst, x0, x1, w0, w1, R *Array, w0Mul, w1Mul, RMul float64) {
	CheckSize(dst.size4D, x0.size4D)
	CheckSize(dst.size4D, x1.size4D)
	C.wavgAsync(
		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(x0.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(x1.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(w0.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(w1.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(R.pointer[0]))),
		C.float(float32(w0Mul)),
		C.float(float32(w1Mul)),
		C.float(float32(RMul)),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))),
		C.int(dst.partLen4D))
	dst.Stream.Sync()
}

// Normalize
func Normalize(m, normMap *Array) {
	C.normalizeAsync(
		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(normMap.pointer[0]))),
		(*C.CUstream)(unsafe.Pointer(&(m.Stream[0]))),
		C.int(m.partLen3D))
	m.Stream.Sync()
}

// Decompose vector to unit vector and length
func Decompose(Mf *Array, m *Array, msat *Array, msatMul float32) {
	C.decomposeAsync(
		(**C.float)(unsafe.Pointer(&(Mf.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(Mf.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(Mf.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),

		(**C.float)(unsafe.Pointer(&(msat.pointer[0]))),

		C.float(msatMul),
		(*C.CUstream)(unsafe.Pointer(&(m.Stream[0]))),
		C.int(m.partLen3D))
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

// Partial dot products (see reduce.h)
func PartialSDot(in1, in2, out *Array, blocks, threadsPerBlock, N int) {
	C.partialSDotAsync(
		(**C.float)(unsafe.Pointer(&in1.pointer[0])),
		(**C.float)(unsafe.Pointer(&in2.pointer[0])),
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

// Partial maximum difference between arrays (see reduce.h)
func PartialMaxSum(a, b, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxSumAsync(
		(**C.float)(unsafe.Pointer(&a.pointer[0])),
		(**C.float)(unsafe.Pointer(&b.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial maximum of Euclidian norm squared (see reduce.h)
func PartialMaxNorm3Sq(x, y, z, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxNorm3SqAsync(
		(**C.float)(unsafe.Pointer(&x.pointer[0])),
		(**C.float)(unsafe.Pointer(&y.pointer[0])),
		(**C.float)(unsafe.Pointer(&z.pointer[0])),
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		C.int(blocks),
		C.int(threadsPerBlock),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Partial maximum of Euclidian norm squared of difference between two 3-vector arrays(see reduce.h)
func PartialMaxNorm3SqDiff(x1, y1, z1, x2, y2, z2, out *Array, blocks, threadsPerBlock, N int) {
	C.partialMaxNorm3SqDiffAsync(
		(**C.float)(unsafe.Pointer(&x1.pointer[0])),
		(**C.float)(unsafe.Pointer(&y1.pointer[0])),
		(**C.float)(unsafe.Pointer(&z1.pointer[0])),
		(**C.float)(unsafe.Pointer(&x2.pointer[0])),
		(**C.float)(unsafe.Pointer(&y2.pointer[0])),
		(**C.float)(unsafe.Pointer(&z2.pointer[0])),
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

func CopyPadZAsync(dst, src *Array, stream Stream) {
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
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

// Padding of a 3D matrix -> only to be used when Ndev=1
// Copy from src to dst, which have different size3D.
// If dst is smaller, the src input is cropped to the right size.
// If dst is larger, the src input is padded with zeros to the right size.
func CopyPad3D(dst, src *Array) {
	Assert(dst.size4D[0] == src.size4D[0] &&
		src.size3D[1] == src.partSize[1] && // only works when Ndev=1
		dst.size3D[1] == dst.partSize[1]) // only works when Ndev=1

	Ncomp := dst.size4D[0]
	D0 := dst.size3D[0]
	D1 := dst.size3D[1]
	D2 := dst.size3D[2]
	S0 := src.size3D[0]
	S1 := src.size3D[1]
	S2 := src.size3D[2]
	C.copyPad3DAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D0),
		C.int(D1),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1),
		C.int(S2),
		C.int(Ncomp),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
	dst.Stream.Sync()
}

func CopyUnPad3D(dst, src *Array) {
	Assert(dst.size4D[0] == src.size4D[0] &&
		src.size3D[1] == src.partSize[1] && // only works when Ndev=1
		dst.size3D[1] == dst.partSize[1]) // only works when Ndev=1

	Ncomp := dst.size4D[0]
	D0 := dst.size3D[0]
	D1 := dst.size3D[1]
	D2 := dst.size3D[2]
	S0 := src.size3D[0]
	S1 := src.size3D[1]
	S2 := src.size3D[2]
	C.copyUnPad3DAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D0),
		C.int(D1),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1),
		C.int(S2),
		C.int(Ncomp),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
	dst.Stream.Sync()
}

func CopyPad3DAsync(dst, src *Array) {
	Assert(dst.size4D[0] == src.size4D[0] &&
		src.size3D[1] == src.partSize[1] && // only works when Ndev=1
		dst.size3D[1] == dst.partSize[1]) // only works when Ndev=1

	Ncomp := dst.size4D[0]
	D0 := dst.size3D[0]
	D1 := dst.size3D[1]
	D2 := dst.size3D[2]
	S0 := src.size3D[0]
	S1 := src.size3D[1]
	S2 := src.size3D[2]
	C.copyPad3DAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D0),
		C.int(D1),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1),
		C.int(S2),
		C.int(Ncomp),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
}

// Insert from src into a block in dst
// E.g.:
// 2x2 src, block = 1, 2x6 dst:
// [ 0 0  S1 S2  0 0 ]
// [ 0 0  S3 S4  0 0 ]
func InsertBlockZ(dst, src *Array, block int) {
	//	AssertMsg(dst.size4D[0] == src.size4D[0], "1")
	//	AssertMsg(dst.size3D[0] == src.size3D[0], "2")
	//	AssertMsg(dst.size3D[1] == src.size3D[1], "3")
	//	AssertMsg(dst.size3D[2] >= src.size3D[2]*(block+1), "4")

	D2 := dst.size3D[2]
	S0 := src.size4D[0] * src.size3D[0] // NComp * Size0
	S1Part := src.partSize[1]
	S2 := src.size3D[2]
	C.insertBlockZAsync(
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

func InsertBlockZAsync(dst, src *Array, block int, stream Stream) {
	//  AssertMsg(dst.size4D[0] == src.size4D[0], "1")
	//  AssertMsg(dst.size3D[0] == src.size3D[0], "2")
	//  AssertMsg(dst.size3D[1] == src.size3D[1], "3")
	//  AssertMsg(dst.size3D[2] >= src.size3D[2]*(block+1), "4")

	D2 := dst.size3D[2]
	S0 := src.size4D[0] * src.size3D[0] // NComp * Size0
	S1Part := src.partSize[1]
	S2 := src.size3D[2]
	C.insertBlockZAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S0),
		C.int(S1Part),
		C.int(S2),
		C.int(block),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

func ZeroArrayAsync(A *Array, stream Stream) {
	N := A.PartLen4D()
	C.zeroArrayAsync(
		(**C.float)(unsafe.Pointer(&A.pointer[0])),
		C.int(N),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

// Extract from src a block to dst
// E.g.:
// 2x2 dst, block = 1, 2x6 src:
// [ 0 0  D1 D2  0 0 ]
// [ 0 0  D3 D4  0 0 ]
func ExtractBlockZ(dst, src *Array, block int) {
	//  AssertMsg(dst.size4D[0] == src.size4D[0], "1")
	//  AssertMsg(dst.size3D[0] == src.size3D[0], "2")
	//  AssertMsg(dst.size3D[1] == src.size3D[1], "3")
	//  AssertMsg(dst.size3D[2]*(block+1) >= src.size3D[2], "4")

	D0 := dst.size4D[0] * dst.size3D[0] // NComp * Size0
	D1Part := dst.partSize[1]
	D2 := dst.size3D[2]
	S2 := src.size3D[2]
	C.extractBlockZAsync(
		(**C.float)(unsafe.Pointer(&dst.pointer[0])),
		C.int(D0),
		C.int(D1Part),
		C.int(D2),
		(**C.float)(unsafe.Pointer(&src.pointer[0])),
		C.int(S2),
		C.int(block),
		(*C.CUstream)(unsafe.Pointer(&(dst.Stream[0]))))
	dst.Stream.Sync()
}

//// Transpose parts on each GPU individually.
func TransposeComplexYZPart(out, in *Array) {
	Assert(
		out.size4D[0] == in.size4D[0] &&
			out.size3D[0] == in.size3D[0] &&
			out.size3D[1]*out.size3D[2] == in.size3D[2]*in.size3D[1])

	C.transposeComplexYZAsyncPart(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.partSize[1]),            //!?
		C.int(in.size3D[2]),              // not / 2 !
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

func TransposeComplexYZPartAsync(out, in *Array, stream Stream) {
	Assert(
		out.size4D[0] == in.size4D[0] &&
			out.size3D[0] == in.size3D[0] &&
			out.size3D[1]*out.size3D[2] == in.size3D[2]*in.size3D[1])

	C.transposeComplexYZAsyncPart(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.partSize[1]),            //!?
		C.int(in.size3D[2]),              // not / 2 !
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

func TransposeComplexYZSingleGPUFWAsync(out, in *Array, stream Stream) {

	C.transposeComplexYZSingleGPUFWAsync(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.partSize[1]),            //!?
		C.int(in.size3D[2]),              // not / 2 !
		C.int(out.size3D[2]),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

func TransposeComplexYZSingleGPUINVAsync(out, in *Array, stream Stream) {

	C.transposeComplexYZSingleGPUINVAsync(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.partSize[1]),            //!?
		C.int(in.size3D[2]),              // not / 2 !
		C.int(out.size3D[1]),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

//this function has only different input for x- and y components
func TransposeComplexYZPart_inv(out, in *Array) {
	Assert(
		out.size4D[0] == in.size4D[0] &&
			out.size3D[0] == in.size3D[0] &&
			out.size3D[1]*out.size3D[2] == in.size3D[2]*in.size3D[1])

	C.transposeComplexYZAsyncPart(
		(**C.float)(unsafe.Pointer(&out.pointer[0])),
		(**C.float)(unsafe.Pointer(&in.pointer[0])),
		C.int(in.size4D[0]*in.size3D[0]), // nComp * N0
		C.int(in.size3D[2])/2,            //!? Why factor 2? -> probably due complex/real.
		C.int(in.partSize[1]*2),          //!? Why factor 2? -> probably due complex/real.
		(*C.CUstream)(unsafe.Pointer(&(out.Stream[0]))))
	out.Stream.Sync()
}

// Computes the uniaxial anisotropy field, stores in h.
func UniaxialAnisotropyAsync(h, m *Array, KuMask, MsatMask *Array, Ku2_Mu0MSat float64, anisUMask *Array, anisUMul []float64, stream Stream) {
	C.uniaxialAnisotropyAsync(
		(**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(KuMask.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(MsatMask.pointer[0]))),
		(C.float)(Ku2_Mu0MSat),
		(**C.float)(unsafe.Pointer(&(anisUMask.Comp[X].pointer[0]))),
		(C.float)(anisUMul[X]),
		(**C.float)(unsafe.Pointer(&(anisUMask.Comp[Y].pointer[0]))),
		(C.float)(anisUMul[Y]),
		(**C.float)(unsafe.Pointer(&(anisUMask.Comp[Z].pointer[0]))),
		(C.float)(anisUMul[Z]),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))),
		(C.int)(h.partLen3D))
}

// 6-neighbor exchange field.
// Aex2_mu0Msatmul: 2 * Aex / Mu0 * Msat.multiplier
func Exchange6Async(h, m, msat, aex *Array, Aex2_mu0Msatmul float64, cellSize []float64, periodic []int, stream Stream) {
	//void exchange6Async(float** hx, float** hy, float** hz, float** mx, float** my, float** mz, float Aex, int N0, int N1Part, int N2, int periodic0, int periodic1, int periodic2, float cellSizeX, float cellSizeY, float cellSizeZ, CUstream* streams);
	CheckSize(h.Size3D(), m.Size3D())
	C.exchange6Async(
		(**C.float)(unsafe.Pointer(&(h.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(h.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[X].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Y].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(m.Comp[Z].pointer[0]))),
		(**C.float)(unsafe.Pointer(&(msat.pointer[0]))),
		(**C.float)(unsafe.Pointer(&(aex.pointer[0]))),
		(C.float)(Aex2_mu0Msatmul),
		(C.int)(h.PartSize()[X]),
		(C.int)(h.PartSize()[Y]),
		(C.int)(h.PartSize()[Z]),
		(C.int)(periodic[X]),
		(C.int)(periodic[Y]),
		(C.int)(periodic[Z]),
		(C.float)(cellSize[X]),
		(C.float)(cellSize[Y]),
		(C.float)(cellSize[Z]),
		(*C.CUstream)(unsafe.Pointer(&(stream[0]))))
}

//// DEBUG: sets all values to their X (i) index
//func SetIndexX(dst *Array) {
//	C.setIndexX(
//		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
//		C.int(dst.size3D[0]),
//		C.int(dst.partSize[1]),
//		C.int(dst.size3D[2]))
//}
//
//// DEBUG: sets all values to their Y (j) index
//func SetIndexY(dst *Array) {
//	C.setIndexY(
//		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
//		C.int(dst.size3D[0]),
//		C.int(dst.partSize[1]),
//		C.int(dst.size3D[2]))
//}
//
//// DEBUG: sets all values to their Z (k) index
//func SetIndexZ(dst *Array) {
//	C.setIndexZ(
//		(**C.float)(unsafe.Pointer(&(dst.pointer[0]))),
//		C.int(dst.size3D[0]),
//		C.int(dst.partSize[1]),
//		C.int(dst.size3D[2]))
//}
