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
	"fmt"
)

type ConvPlan struct {
	dataSize  [3]int   // Size of the (non-zero) input data block
	logicSize [3]int   // Non-transformed kernel size >= dataSize
	fftKern   [6]Array // transformed kernel components, unused ones are nil.
	fftIn     Array    // transformed input data
	fft       FFTPlan  // transforms input/output data
}

func (conv *ConvPlan) Init(dataSize []int, kernel []*host.Array) {
	Assert(len(dataSize) == 3)
	Assert(len(kernel) == 6)

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
		//conv.storeSize[i] = kernSize[i]
	}

	// init fft
	conv.fftIn.Init(1, []int{logicSize[0], logicSize[1], logicSize[2] + 2}, DO_ALLOC)
	conv.fft.Init(dataSize, logicSize)

	Debug("ConvPlan.init", "dataSize:", conv.dataSize, "logicSize:", conv.logicSize)

	// init fftKern
	for i, k := range kernel {
		if k != nil {
			Debug("ConvPlan.init", "alloc:", kernString[i])
			conv.fftKern[i].Init(1, []int{logicSize[0], logicSize[1], logicSize[2]/2 + 1}, DO_ALLOC) // not so aligned..
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
// This saves a huge amount of memory
func (conv *ConvPlan) loadKernel(kernel []*host.Array) {

	// Check sanity of kernel
	for i, k := range kernel {
		if k != nil {
			for _, e := range k.List {
				AssertMsg(IsReal(e), "K", kernString[i], "=", e)
			}
		}
	}

	var fft FFTPlan
	defer fft.Free()
	fft.Init(conv.logicSize[:], conv.logicSize[:])

	logic := conv.logicSize[:]
	devIn := NewArray(1, logic)
	defer devIn.Free()
	devOut := NewArray(1, []int{logic[0], logic[1], logic[2] + 2}) // +2 elements: R2C
	defer devOut.Free()

	for i, k := range kernel {
		if k != nil {
			fmt.Println("kern", kernString[i], kernel[i].Array)
			devIn.CopyFromHost(k)
			fft.Forward(devIn, devOut)
			scaleRealParts(&conv.fftKern[i], devOut, 1/float32(fft.Normalization()))
			fmt.Println("fftKern", kernString[i], conv.fftKern[i].LocalCopy().Array)
		}
	}

}

//func NewConvPlan(dataSize,  []int) *ConvPlan {
//	conv := new(ConvPlan)
//	conv.Init(dataSize, kernSize)
//	return conv
//}

// Extract real parts, copy them from src to dst.
// In the meanwhile, check if imaginary parts are nearly zero.
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
	for c := range in.Comp {
		conv.fft.Forward(&in.Comp[c], &conv.fftIn)
	}
}

// indices for (anti-)symmetric kernel when only 6 of the 9 components are stored.
const (
	XX = 0
	YY = 1
	ZZ = 2
	YZ = 3
	XZ = 4
	XY = 5
)

var kernString map[int]string = map[int]string{XX: "XX", YY: "YY", ZZ: "ZZ", YZ: "YZ", XZ: "XZ", XY: "XY"}
