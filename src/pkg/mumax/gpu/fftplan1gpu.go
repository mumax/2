//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	//"cuda/cufft"
	//"fmt"
)

// Specialized FFT plan for 1 GPU,
// more efficient than the general FFT on 1GPU
type FFTPlan1GPU struct {
	nComp    int    // Number of components
	dataSize [3]int // Size of the (non-zero) input data block
	fftSize  [3]int // Transform size including zero-padding. >= dataSize
}

func (fft *FFTPlan1GPU) Init(nComp int, dataSize, fftSize []int) {
	Assert(NDevice() == 1)

	// init size
	fft.nComp = nComp
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.fftSize[i] = fftSize[i]
	}
}

func NewFFTPlan1GPU(nComp int, dataSize, fftSize []int) *FFTPlan1GPU {
	fft := new(FFTPlan1GPU)
	fft.Init(nComp, dataSize, fftSize)
	return fft
}

func (fft *FFTPlan1GPU) Free() {
	fft.nComp = 0
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.fftSize[i] = 0
	}
	// TODO destroy cufft plan
	// TODO free buffers
}

func (fft *FFTPlan1GPU) Forward(in, out *Array) {
}
