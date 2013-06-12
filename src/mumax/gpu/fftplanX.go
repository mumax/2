//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Authors: Mykola Dvornik
// has no multi-gpu support

import (
	"cuda/cufft"
	. "mumax/common"
)

//Register this FFT plan
func init() {
	fftPlans["X"] = NewFFTPlanX
}

type FFTPlanX struct {
	//sizes
	dataSize   [3]int // Size of the (non-zero) input data block
	logicSize  [3]int // Transform size including zero-padding. >= dataSize
	outputSize [3]int // Size of the output data (one extra row PER GPU)

	// fft plans
	plan3D_FWD cufft.Handle
	plan3D_BWD cufft.Handle
	Stream                   
}

func (fft *FFTPlanX) init(dataSize, logicSize []int) {
	if NDevice() > 1 {
		panic(InputErrF("FFT Plan X has no multi-gpu support."))
	}
	
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)
	const nComp = 1
	
	Debug(dataSize, logicSize)

	outputSize := FFTOutputSize(logicSize)
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.logicSize[i] = logicSize[i]
		fft.outputSize[i] = outputSize[i]
	}

	fft.Stream = NewStream()

	fft.plan3D_FWD = cufft.Plan3d(fft.logicSize[0], fft.logicSize[1], fft.logicSize[2], cufft.R2C)
	fft.plan3D_FWD.SetStream(uintptr(fft.Stream[0]))
	fft.plan3D_FWD.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	fft.plan3D_BWD = cufft.Plan3d(fft.logicSize[0], fft.logicSize[1], fft.logicSize[2], cufft.C2R)
	fft.plan3D_BWD.SetStream(uintptr(fft.Stream[0]))
	fft.plan3D_BWD.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
}

func NewFFTPlanX(dataSize, logicSize []int) FFTInterface {
	fft := new(FFTPlanX)
	fft.init(dataSize, logicSize)
	return fft
}

func (fft *FFTPlanX) Free() {
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.logicSize[i] = 0
	}
}

func (fft *FFTPlanX) Forward(in, out *Array) {
	AssertMsg(in.size4D[0] == 1, "1")
	AssertMsg(out.size4D[0] == 1, "2")
	CheckSize(in.size3D, fft.dataSize[:])
	CheckSize(out.size3D, fft.outputSize[:])
	
	CopyPad3D(out, in)
	lin := in.LocalCopy()
	lout := out.LocalCopy()
	Debug("in: ", lin.Array)
	Debug("out: ", lout.Array)
	ptr := uintptr(out.pointer[0]) 
	fft.plan3D_FWD.ExecR2C(ptr, ptr)
	fft.Sync() //  Is this required?
	
	llout := out.LocalCopy()
	Debug("out: ", llout.Array)
}

func (fft *FFTPlanX) Inverse(in, out *Array) {
	
	ptr := uintptr(in.pointer[0]) 
	fft.plan3D_BWD.ExecC2R(ptr, ptr)
	
	fft.Sync() //  Is this required?
	// extracting data
	CopyPad3D(out, in)
	
	lin := in.LocalCopy()
	lout := out.LocalCopy()
	Debug("in: ", lin.Array)
	Debug("out: ", lout.Array)
}
