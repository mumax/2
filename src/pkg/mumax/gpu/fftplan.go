//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	"cuda/cufft"
	"fmt"
)

type FFTPlan struct {
	nComp    int
	dataSize [3]int         // Size of the (non-zero) input data block
	fftSize  [3]int         // Transform size including zero-padding. >= dataSize
	padZ     Array          // Buffer for Z-padding and +2 elements
	planZ    []cufft.Handle // In-place transform of padZ parts, 1/GPU /// ... from outer space
	transp1  Array          // Buffer for partial transpose per GPU
	transp2  Array          // Buffer for full transpose+padY, incl. inter-GPU swaps
	Stream                  //
}

func (fft *FFTPlan) Init(nComp int, dataSize, fftSize []int) {
	// init size
	fft.nComp = nComp
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.fftSize[i] = fftSize[i]
	}

	// init stream
	fft.Stream = NewStream()

	// init padZ
	padZN0 := fft.dataSize[0]
	padZN1 := fft.dataSize[1]
	padZN2 := fft.fftSize[2] + 2
	fft.padZ.Init(nComp, []int{padZN0, padZN1, padZN2}, DO_ALLOC)

	// init planZ
	fft.planZ = make([]cufft.Handle, NDevice())
	for dev := range _useDevice {
		setDevice(_useDevice[dev])
		fft.planZ[dev] = cufft.Plan1d(fft.fftSize[2], cufft.R2C, (nComp*padZN0*padZN1)/NDevice())
		fft.planZ[dev].SetStream(uintptr(fft.Stream[dev]))
	}

	// init transp1
	fft.transp1.Init(nComp, fft.padZ.size3D, DO_ALLOC)

	// init transp2
	t2N0 := dataSize[0]
	t2N1 := ((fftSize[2] / NDevice()) + 2) * NDevice()
	t2N2 := dataSize[1]
	fft.transp2.Init(nComp, []int{t2N0, t2N1, t2N2}, DO_ALLOC)
}

func NewFFTPlan(nComp int, dataSize, fftSize []int) *FFTPlan {
	fft := new(FFTPlan)
	fft.Init(nComp, dataSize, fftSize)
	return fft
}

func (fft *FFTPlan) Free() {
	fft.nComp = 0
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.fftSize[i] = 0
	}
	(&(fft.padZ)).Free()

	// TODO destroy
}

func (fft *FFTPlan) Forward(in, out *Array) {
	padZ := fft.padZ
	transp1 := fft.transp1

	fmt.Println("in:", in.LocalCopy().Array)

	CopyPadZ(&(padZ), in)
	fmt.Println("padZ:", padZ.LocalCopy().Array)

	//for dev := range _useDevice {
	//	fft.planZ[dev].ExecR2C(uintptr(padZ.pointer[dev]), uintptr(padZ.pointer[dev])) // is this really async?
	//}
	//fft.Sync()
	fmt.Println("fftZ:", padZ.LocalCopy().Array)

	TransposeComplexYZPart(&transp1, &padZ)
	fmt.Println("transp1:", transp1.LocalCopy().Array)
}
