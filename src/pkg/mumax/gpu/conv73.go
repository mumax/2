//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// 7-input, 3 output Convolution plan for general Maxwell equations.
// This is a basic implementation whose memory bandwidth utilization can be improved.
//
// 7 sources:
// 1 charge (electric or magnetic)
// 3 dipole components (electric polarization or magnetization)
// 3 current components (electric or displacement)
//
// 3 kernels:
// MONOPOLE (1x3) for Gauss law on monopole sources
// DIPOLE (3x3, symmetric) for Gauss law on dipole sources
// ROTOR (3x3, anitsymmetric (?)) for Faraday/Ampère law on current sources
//
// 3 
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"rand"
	"runtime"
)


const (
	Nin  = 7
	Nout = 3
)

// Full Maxwell convolution plan
type Conv73Plan struct {
	dataSize    [3]int               // Size of the (non-zero) input data block
	logicSize   [3]int               // Non-transformed kernel size >= dataSize
	fftKern     [Nin][Nout]*Array    // transformed kernel non-redundant parts (only real or imag parts, or nil)
	fftMul      [Nin][Nout]complex64 // multipliers for kernel
	fftBuffer   Array                // transformed input data
	fftOut      [Nout]*Array         // transformed output data
	fftPlan     FFTInterface         // transforms input/output data
	fullFFTPlan FFTInterface         // transforms kernel // TODO: free?
}


func NewConv73Plan(dataSize, logicSize []int) *Conv73Plan {
	conv := new(Conv73Plan)
	conv.Init(dataSize, logicSize)
	return conv
}

// Kernel does not need to take into account unnormalized FFTs,
// this is handled by the convplan.
func (conv *Conv73Plan) Init(dataSize, logicSize []int) {
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)

	conv.Free() // must not leak memory on 2nd init.

	// init size
	for i := range conv.dataSize {
		conv.dataSize[i] = dataSize[i]
		conv.logicSize[i] = logicSize[i]
	}

	// init fft
	fftOutputSize := FFTOutputSize(logicSize)
	conv.fftBuffer.Init(1, fftOutputSize, DO_ALLOC) // TODO: recycle
	for i := 0; i < Nout; i++ {
		conv.fftOut[i].Init(1, fftOutputSize, DO_ALLOC) // TODO: recycle
	}
	conv.fftPlan = NewDefaultFFT(dataSize, logicSize)

	// init fftKern
	fftKernSize := FFTOutputSize(logicSize)
	fftKernSize[2] = fftKernSize[2] / 2 // store only non-redundant parts

}


func (conv *Conv73Plan) LoadKernel(kernel *host.Array, pos int, matsymm int, realness int) {
	//Assert(kernel.NComp() == 9) // full tensor
	Assert(matsymm == MatrixSymmetry(kernel))
	Assert(matsymm == SYMMETRIC || matsymm == ANTISYMMETRIC || matsymm == NOSYMMETRY || matsymm == DIAGONAL)

	//if FFT'd kernel is pure real or imag, 
	//store only relevant part and multiply by scaling later
	scaling := [3]complex64{complex(1, 0), complex(0, 1), complex(0, 0)}[realness]
	Debug("scaling=", scaling)

	// FFT input on GPU
	logic := conv.logicSize[:]
	devIn := NewArray(1, logic)
	defer devIn.Free()

	// FFT output on GPU
	devOut := NewArray(1, FFTOutputSize(logic))
	defer devOut.Free()

	// FFT all components
	fftKern := conv.fftKern
	fftMul := conv.fftMul
	for k := 0; k < 9; k++ {
		i, j := IdxToIJ(k) // fills diagonal first, then upper, then lower

		// clear first
		fftKern[i+pos][j] = nil
		fftMul[i+pos][j] = 0

		// ignore zeros
		if k < kernel.NComp() && IsZero(kernel.Comp[k]) {
			Debug("kernel", TensorIndexStr[k], " == 0")
			continue
		}

		// ignore off-diagonals of vector (would go out of bounds)
		if k > ZZ && matsymm == DIAGONAL {
			continue
		}

		// handle lower triangle
		if k > XY {
			if matsymm == SYMMETRIC {
				fftKern[i+pos][j] = fftKern[j+pos][i]
				fftMul[i+pos][j] = fftMul[j+pos][i]
				continue
			}
			if matsymm == ANTISYMMETRIC {
				fftKern[i+pos][j] = fftKern[j+pos][i]
				fftMul[i+pos][j] = -fftMul[j+pos][i]
				continue
			}
		}

		// normal case

		devIn.CopyFromHost(kernel.Component(k))
		conv.fullFFTPlan.Forward(devIn, devOut)
		hostOut := devOut.LocalCopy()

		hostFFTKern := extract(hostOut, realness)
		rescale(hostFFTKern, 1/float64(FFTNormLogic(logic)))
		fftKern[j+pos][i].CopyFromHost(hostFFTKern)
	}
}


func IsZero(array []float32) bool {
	for _, x := range array {
		if x != 0 {
			return false
		}
	}
	return true
}


// arr[i] *= scale
func rescale(arr *host.Array, scale float64) {
	list := arr.List
	for i := range list {
		list[i] = float32(float64(list[i]) * scale)
	}
}

// matrix symmetry
const (
	NOSYMMETRY    = 0  // Kij independent of Kji
	SYMMETRIC     = 1  // Kij = Kji
	DIAGONAL      = 2  // also used for vector
	ANTISYMMETRIC = -1 // Kij = -Kji
)

// Detects matrix symmetry.
// returns NOSYMMETRY, SYMMETRIC, ANTISYMMETRIC 
func MatrixSymmetry(matrix *host.Array) int {
	AssertMsg(matrix.NComp() == 9, "MatrixSymmetry NComp")
	symm := true
	asymm := true
	for i := 0; i < Nout; i++ {
		for j := 0; j < Nout; j++ {
			idx1 := TensorIdx[i][j]
			idx2 := TensorIdx[j][i]
			comp1 := matrix.Comp[idx1]
			comp2 := matrix.Comp[idx2]
			for x := range comp1 {
				if comp1[x] != comp2[x] {
					symm = false
					if !asymm {
						break
					}
				}
				if comp1[x] != -comp2[x] {
					asymm = false
					if !symm {
						break
					}
				}
			}
		}
	}
	if symm {
		return SYMMETRIC // also covers all zeros
	}
	if asymm {
		return ANTISYMMETRIC
	}
	return NOSYMMETRY
}


// data realness
const (
	PUREREAL = 0 // data is purely real
	PUREIMAG = 1 // data is purely complex
	COMPLEX  = 2 // data is full complex number
)

// Detects realness of complex data (interleaved format).
// returns 

// INTERNAL: Loads a convolution kernel.
// This is automatically done during initialization.
// "kernel" is not FFT'ed yet, this is done here.
// We use exactly the same fft as for the magnetizaion
// so that the convolution definitely works.
// After FFT'ing, the kernel is purely real,
// so we discard the imaginary parts.
// This saves a huge amount of memory.
// The kernel is internally scaled to compensate
// for unnormalized FFTs
func loadKernComp(fftKern *Array, fft FFTInterface, kernel *host.Array, op int) {

	// Check sanity of kernel
	//	for _, e := range kernel.List {
	//		if !IsReal(e) {
	//			BugF("Kern", opStr[op], "=", e)
	//		}
	//	}

	//logic := kernel.Size3D
	//devIn := NewArray(1, logic)
	//defer devIn.Free()

	//devOut := NewArray(1, FFTOutputSize(logic))
	//defer devOut.Free()

	//devIn.CopyFromHost(kernel)
	//fft.Forward(devIn, devOut)
	//scalePart(fftKern, devOut, 1/float32(FFTNormLogic(logic)), oddness[op])

}


func (conv *Conv73Plan) Free() {
	// TODO
}

func (conv *Conv73Plan) Convolve(in, out *Array) {
	//		fftBuf := &conv.fftBuf
	//		for i:=MONOPOLE; i<=ROTOR; i++{
	//
	//		fftKern := &conv.fftKern
	//	
	//		conv.ForwardFFT(in)
	//	
	//		// Point-wise kernel multiplication
	//		KernelMulMicromag3DAsync(&fftIn.Comp[X], &fftIn.Comp[Y], &fftIn.Comp[Z],
	//			fftKern[XX], fftKern[YY], fftKern[ZZ],
	//			fftKern[YZ], fftKern[XZ], fftKern[XY],
	//			fftIn.Stream) // TODO: choose stream wisely
	//		fftIn.Stream.Sync() // !!
	//	
	//		conv.InverseFFT(out)
}

// 	INTERNAL
// Sparse transform all 3 components.
// (FFTPlan knows about zero padding etc)
func (conv *Conv73Plan) ForwardFFT(in *Array) {
	//	for c := range in.Comp {
	//		conv.fft.Forward(&in.Comp[c], &conv.fftIn.Comp[c])
	//	}
}

// 	INTERNAL
// Sparse backtransform
// (FFTPlan knows about zero padding etc)
func (conv *Conv73Plan) InverseFFT(out *Array) {
	//	for c := range out.Comp {
	//		conv.fft.Inverse(&conv.fftIn.Comp[c], &out.Comp[c])
	//	}
}

func (conv *Conv73Plan) SelfTest() {
	Debug("FFT self-test")
	rng := rand.New(rand.NewSource(0))
	size := conv.dataSize[:]

	in := NewArray(1, size)
	defer in.Free()
	arr := in.LocalCopy()
	a := arr.List
	for i := range a {
		a[i] = 2*rng.Float32() - 1
		if a[i] == 0 {
			a[i] = 1
		}
	}
	in.CopyFromHost(arr)

	out := NewArray(1, size)
	defer out.Free()

	conv.ForwardFFT(in)
	conv.InverseFFT(out)

	b := out.LocalCopy().List
	norm := float32(1 / float64(FFTNormLogic(conv.logicSize[:])))
	var maxerr float32
	for i := range a {
		if Abs32(a[i]-b[i]*norm) > maxerr {
			maxerr = Abs32(a[i] - b[i]*norm)
		}
	}
	Debug("FFT max error:", maxerr)
	if maxerr > 1e-3 {
		panic(BugF("FFT self-test failed, max error:", maxerr, "\nPlease use a different grid size of FFT type."))
	}
	runtime.GC()
}

const (
	REAL = 0
	IMAG = 1
)


// Extract real or imaginary parts, copy them from src to dst.
// In the meanwhile, check if the other parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
// real_imag = 0: real parts
// real_imag = 1: imag parts
func extract(src *host.Array, realness int) *host.Array {
	if realness == COMPLEX {
		return src
	}

	sx := src.Size3D[X]
	sy := src.Size3D[Y]
	sz := src.Size3D[Z] / 2 // only real/imag parts
	dst := host.NewArray(src.NComp(), []int{sx, sy, sz})

	dstList := dst.List
	srcList := src.List

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maxbad := float32(0.)
	maxgood := float32(0.)
	other := 1 - realness
	for i := range dstList {
		dstList[i] = srcList[2*i+realness]
		if Abs32(srcList[2*i+other]) > maxbad {
			maxbad = Abs32(srcList[2*i+other])
		}
		if Abs32(srcList[2*i+realness]) > maxgood {
			maxgood = Abs32(srcList[2*i+realness])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max part", realness, ":", maxgood)
	Debug("FFT Kernel max part", other, ":", maxbad)
	Debug("FFT Kernel max bad/good part=", maxbad/maxgood)
	if maxbad/maxgood > 1e-5 { // TODO: is this reasonable?
		Warn("FFT Kernel max bad/good part=", maxbad/maxgood)
	}
	return dst
}
