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
	//	cu "cuda/driver"
	//	"cuda/cufft"
	//	"fmt"
)

type ConvPlan struct {
	dataSize [3]int    // Size of the (non-zero) input data block
	logicSize [3]int   // Non-transformed kernel size >= dataSize
	storeSize [3]int   // transformed kernel size (logic size + 2 Z elements)
	fftKern  [6]Array // transformed kernel components, unused ones are nil.
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

func (conv *ConvPlan) Init(dataSize []int, kernel []*host.Array) {
	Assert(len(dataSize) == 3)
	Assert(len(kernel) == 6)

	// find non-nil kernel element to get kernel size
	var kernSize []int
	for _, k := range kernel {
		if k != nil {
			kernSize = k.Size3D
			break
		}
	}

	// init size
	for i := range conv.dataSize {
		conv.dataSize[i] = dataSize[i]
		conv.logicSize[i] = kernSize[i]
		conv.storeSize[i] = kernSize[i]
	}
	conv.storeSize[Z] += 2 // 2 extra elements due to R2C FFT

	Debug("ConvPlan.init", "dataSize:", conv.dataSize, "logicSize:", conv.logicSize, "storeSize:", conv.storeSize)

	// init fftKern
	for i, k := range kernel {
		if k != nil {
			Debug("ConvPlan.init", "alloc:", kernString[i])
			conv.fftKern[i].Init(1, kernSize, DO_ALLOC)
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
	fft.Init(conv.dataSize[:], conv.logicSize[:])
	//norm := 1 / float64(fft.Normalization())

	devIn := NewArray(1, conv.logicSize[:])
	defer devIn.Free()
	devOut := NewArray(1, conv.storeSize[:])
	defer devOut.Free()

	for i,k := range kernel{
		if k != nil{
			devIn.CopyFromHost(k)	
			fft.Forward(devIn, devOut)
			CopyRealPart(conv.fftKern[i], devOut)	
		}
	}

//	hostOut := tensor.NewT3(fft.PhysicSize())
//
//	//   allocCount := 0
//
//	for i := range conv.kernel {
//		if conv.needKernComp(i) { // the zero components are not stored
//			//       allocCount++
//			TensorCopyTo(kernel[i], devIn)
//			fft.Forward(devIn, devOut)
//			TensorCopyFrom(devOut, hostOut)
//			listOut := hostOut.List()
//
//			// Normally, the FFT'ed kernel is purely real because of symmetry,
//			// so we only store the real parts...
//			maximg := float32(0.)
//			for j := 0; j < len(listOut)/2; j++ {
//				listOut[j] = listOut[2*j] * norm
//				if abs32(listOut[2*j+1]) > maximg {
//					maximg = abs32(listOut[2*j+1])
//				}
//			}
//			// ...however, we check that the imaginary parts are nearly zero,
//			// just to be sure we did not make a mistake during kernel creation.
//			if maximg > 1e-4 {
//				fmt.Fprintln(os.Stderr, "Warning: FFT Kernel max imaginary part=", maximg)
//			}
//
//			conv.kernel[i] = NewTensor(conv.Backend, conv.KernelSize())
//			conv.memcpyTo(&listOut[0], conv.kernel[i].data, Len(conv.kernel[i].Size()))
//		}
//	}
//
//	//   fmt.Println(allocCount, " non-zero kernel components.")
//	fft.Free()
//	devIn.Free()
//	devOut.Free()

}
//func NewConvPlan(dataSize,  []int) *ConvPlan {
//	conv := new(ConvPlan)
//	conv.Init(dataSize, kernSize)
//	return conv
//}

func (conv *ConvPlan) Free() {
	// TODO
}

func (conv *ConvPlan) Convolve(in, out *Array) {

}
