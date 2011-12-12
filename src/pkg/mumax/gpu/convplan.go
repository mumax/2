//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Convolution plan
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
)

// Convolution plan
type ConvPlan struct {
	dataSize  [3]int       // Size of the (non-zero) input data block
	logicSize [3]int       // Non-transformed kernel size >= dataSize
	fftKern   [6]*Array    // transformed kernel components, unused ones are nil.
	fftIn     Array        // transformed input data
	fft       FFTInterface // transforms input/output data
}

// Kernel does not need to take into account unnormalized FFTs,
// this is handled by the convplan.
func (conv *ConvPlan) Init(dataSize []int, kernel []*host.Array, fftKern *Array) {
	Assert(len(dataSize) == 3)
	Assert(len(kernel) == 6)

	conv.Free() // must not leak memory on 2nd init.

	// find non-nil kernel element to get kernel size
	var logicSize []int
	for _, k := range kernel {
		if k != nil {
			logicSize = k.Size3D
			break
		}
	}

	// init size
	for i := range conv.dataSize {
		conv.dataSize[i] = dataSize[i]
		conv.logicSize[i] = logicSize[i]
	}

	// init fft
	conv.fftIn.Init(3, []int{logicSize[0], logicSize[1], logicSize[2] + 2*NDevice()}, DO_ALLOC) // TODO: FFTPlan.OutputSize()


	conv.fft = NewDefaultFFT(dataSize, logicSize)

	Debug("ConvPlan.init", "dataSize:", conv.dataSize, "logicSize:", conv.logicSize)

	// init fftKern
	//	for i, k := range kernel {
	//		if k != nil {
	//			Debug("ConvPlan.init", "alloc:", TensorIndexStr[i])
	//			conv.fftKern[i].Init(1, []int{logicSize[0], logicSize[1], logicSize[2]/2 + 1}, DO_ALLOC) // not so aligned..
	//		}
	//	}

	fftKernSize := []int{logicSize[0], logicSize[1], (logicSize[2] + 2*NDevice()) / 2}
	CheckSize(fftKernSize, fftKern.Size3D())
	for i, k := range kernel {
		if k != nil {
			Debug("ConvPlan.init", "use K", TensorIndexStr[i])
			conv.fftKern[i] = &fftKern.Comp[i]
		}
	}

	conv.loadKernel(kernel)

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
func (conv *ConvPlan) loadKernel(kernel []*host.Array) {

	// Check sanity of kernel
	for i, k := range kernel {
		if k != nil {
			for _, e := range k.List {
				AssertMsg(IsReal(e), "K", TensorIndexStr[i], "=", e)
			}
		}
	}

	fft := NewDefaultFFT(conv.logicSize[:], conv.logicSize[:])
	defer fft.Free()

	logic := conv.logicSize[:]
	devIn := NewArray(1, logic)
	defer devIn.Free()
	devOut := NewArray(1, []int{logic[0], logic[1], logic[2] + 2*NDevice()}) // +2 elements: R2C
	defer devOut.Free()

	for i, k := range kernel {
		if k != nil {
			//fmt.Println("kern", TensorIndexStr[i], kernel[i].Array)
			devIn.CopyFromHost(k)
			fft.Forward(devIn, devOut)
			scaleRealParts(conv.fftKern[i], devOut, 1/float32(FFTNormLogic(logic)))
			//fmt.Println("fftKern", TensorIndexStr[i], conv.fftKern[i].LocalCopy().Array)
		}
	}

}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
func scaleRealParts(dst, src *Array, scale float32) {
	Assert(dst.size3D[0] == src.size3D[0] &&
		dst.size3D[1] == src.size3D[1] &&
		dst.size3D[2] == src.size3D[2]/2)

	dstHost := dst.LocalCopy()
	srcHost := src.LocalCopy()
	dstList := dstHost.List
	srcList := srcHost.List

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maximg := float32(0.)
	for i := range dstList {
		dstList[i] = srcList[2*i] * scale
		if Abs32(srcList[2*i+1]) > maximg {
			maximg = Abs32(srcList[2*i+1])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max imaginary part=", maximg)
	if maximg*scale > 1e-5 { // TODO: is this reasonable? How about 
		Warn("FFT Kernel max imaginary part=", maximg)
	}

	dst.CopyFromHost(dstHost)
}

func (conv *ConvPlan) Free() {
	// TODO
}

func (conv *ConvPlan) Convolve(in, out *Array) {
	//Debug("Convolve")
	fftIn := &conv.fftIn
	fftKern := &conv.fftKern

	// First transform all 3 components 
	// (FFTPlan knows about zero padding etc)
	for c := range in.Comp {
		conv.fft.Forward(&in.Comp[c], &fftIn.Comp[c])
	}

	// Point-wise kernel multiplication
	KernelMulMicromag3DAsync(&fftIn.Comp[X], &fftIn.Comp[Y], &fftIn.Comp[Z],
		fftKern[XX], fftKern[YY], fftKern[ZZ],
		fftKern[YZ], fftKern[XZ], fftKern[XY],
		fftIn.Stream) // TODO: choose stream wisely

	// Backtransform
	// (FFTPlan knows about zero padding etc)
	for c := range in.Comp {
		conv.fft.Inverse(&conv.fftIn.Comp[c], &out.Comp[c])
	}
}
