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

// Full Maxwell convolution plan
type Conv73Plan struct {
	dataSize  [3]int       // Size of the (non-zero) input data block
	logicSize [3]int       // Non-transformed kernel size >= dataSize
	fftKern   [][]*Array   // MONOPOLE, DIPOLE and ROTOR kernels
	fftBuffer Array        // transformed input data
	fftPlan   FFTInterface // transforms input/output data
}

// index for kind of kernel
const (
	MONOPOLE = 0
	DIPOLE   = 1
	ROTOR    = 2
)

var opStr map[int]string = map[int]string{MONOPOLE: "MONOPOLE", DIPOLE: "DIPOLE", ROTOR: "ROTOR"}


func NewConv73Plan(dataSize []int, kernMono, kernDi, kernRot []*host.Array) *Conv73Plan {
	conv := new(Conv73Plan)
	conv.Init(dataSize, kernMono, kernDi, kernRot)
	return conv
}

// stores whether the real or imaginary part of a kernel type should be used.
var oddness map[int]int = map[int]int{MONOPOLE: IMAG, DIPOLE: REAL} //TODO: rotor?

// Kernel does not need to take into account unnormalized FFTs,
// this is handled by the convplan.
func (conv *Conv73Plan) Init(dataSize []int, kernMono, kernDi, kernRot []*host.Array) {
	Assert(len(dataSize) == 3)

	conv.Free() // must not leak memory on 2nd init.

	// find non-nil kernel element to get kernel size
	kernels := [][]*host.Array{kernMono, kernDi, kernRot}
	var logicSize []int
	for _, kern := range kernels {
		if kern == nil {
			continue
		}
		for _, k := range kern {
			if k != nil {
				logicSize = k.Size3D
				break
			}
		}
	}

	// init size
	for i := range conv.dataSize {
		conv.dataSize[i] = dataSize[i]
		conv.logicSize[i] = logicSize[i]
	}

	// init fft
	fftOutputSize := FFTOutputSize(logicSize)
	conv.fftBuffer.Init(1, fftOutputSize, DO_ALLOC)
	conv.fftPlan = NewDefaultFFT(dataSize, logicSize)

	// init fftKern
	fftKernSize := FFTOutputSize(logicSize)
	fftKernSize[2] = fftKernSize[2] / 2 // store only non-redundant parts
	conv.fftKern = make([][]*Array, ROTOR)

	// transforms the kernel, FFT is not sparse
	fullFFTPlan := NewDefaultFFT(conv.logicSize[:], conv.logicSize[:])
	defer fullFFTPlan.Free()

	for op, kernel := range kernels {
		if kernel == nil {
			continue
		}
		conv.fftKern[op] = make([]*Array, len(kernel))
		for i, k := range kernel {
			if k != nil {
				Debug("Conv73Plan.init", "use K", opStr[op], TensorIndexStr[i])
				conv.fftKern[op][i] = NewArray(1, fftKernSize)
				CheckSize(fftKernSize, conv.fftKern[op][i].Size3D())
				loadKernComp(conv.fftKern[op][i], fullFFTPlan, kernel[i], op)
			}
		}
	}
	runtime.GC()
}


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
	for _, e := range kernel.List {
		if !IsReal(e) {
			BugF("Kern", opStr[op], "=", e)
		}
	}

	logic := kernel.Size3D
	devIn := NewArray(1, logic)
	defer devIn.Free()

	devOut := NewArray(1, FFTOutputSize(logic))
	defer devOut.Free()

	devIn.CopyFromHost(kernel)
	fft.Forward(devIn, devOut)
	scalePart(fftKern, devOut, 1/float32(FFTNormLogic(logic)), oddness[op])

}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
//func scaleRealParts(dst, src *Array, scale float32) {
//	Assert(dst.size3D[0] == src.size3D[0] &&
//		dst.size3D[1] == src.size3D[1] &&
//		dst.size3D[2] == src.size3D[2]/2)
//
//	dstHost := dst.LocalCopy()
//	srcHost := src.LocalCopy()
//	dstList := dstHost.List
//	srcList := srcHost.List
//
//	// Normally, the FFT'ed kernel is purely real because of symmetry,
//	// so we only store the real parts...
//	maximg := float32(0.)
//	for i := range dstList {
//		dstList[i] = srcList[2*i] * scale
//		if Abs32(srcList[2*i+1]) > maximg {
//			maximg = Abs32(srcList[2*i+1])
//		}
//	}
//	// ...however, we check that the imaginary parts are nearly zero,
//	// just to be sure we did not make a mistake during kernel creation.
//	Debug("FFT Kernel max imaginary part=", maximg)
//	if maximg*scale > 1 { // TODO: is this reasonable?
//		Warn("FFT Kernel max imaginary part=", maximg)
//	}
//
//	dst.CopyFromHost(dstHost)
//}

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
func scalePart(dst, src *Array, scale float32, real_imag int) {
	Assert(real_imag == 0 || real_imag == 1)
	Assert(dst.size3D[0] == src.size3D[0] &&
		dst.size3D[1] == src.size3D[1] &&
		dst.size3D[2] == src.size3D[2]/2)

	dstHost := dst.LocalCopy()
	srcHost := src.LocalCopy()
	dstList := dstHost.List
	srcList := srcHost.List

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maxbad := float32(0.)
	maxgood := float32(0.)
	other := 1 - real_imag
	for i := range dstList {
		dstList[i] = srcList[2*i+real_imag] * scale
		if Abs32(srcList[2*i+other]) > maxbad {
			maxbad = Abs32(srcList[2*i+other])
		}
		if Abs32(srcList[2*i+real_imag]) > maxgood {
			maxgood = Abs32(srcList[2*i+real_imag])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max part", real_imag, ":", maxgood)
	Debug("FFT Kernel max part", other, ":", maxbad)
	Debug("FFT Kernel max bad/good part=", maxbad/maxgood)
	if maxbad/maxgood > 1e-5 { // TODO: is this reasonable?
		Warn("FFT Kernel max bad/good part=", maxbad/maxgood)
	}

	dst.CopyFromHost(dstHost)
}
