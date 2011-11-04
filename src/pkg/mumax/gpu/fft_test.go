//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// DO NOT USE TEST.FATAL: -> runtime.GoExit -> context switch -> INVALID CONTEXT!

package gpu

// Author: Arne Vansteenkiste

import (
	"testing"
)

func TestFFT(test *testing.T) {
	nComp := 1
	dataSize := []int{1, 2, 8}
	fftSize := []int{1, 2, 8}
	fft := NewFFTPlan(nComp, dataSize, fftSize)
	defer fft.Free()

	in := NewArray(nComp, dataSize)
	defer in.Free()
	inh := in.LocalCopy()
	for i := range inh.List {
		inh.List[i] = float32(i)
	}
	//inh.List[0] = 1
	in.CopyFromHost(inh)

	fft.Forward(in, nil)

}
