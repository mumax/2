//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	"fmt"
)

type FFTPlan struct {
	nComp    int
	dataSize [3]int // Size of the (non-zero) input data block
	fftSize  [3]int // Transform size including zero-padding. >= dataSize
	padZ     Array
}

func (fft *FFTPlan) Init(nComp int, dataSize3D, fftSize3D []int) {
	fft.nComp = nComp
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize3D[i]
		fft.fftSize[i] = fftSize3D[i]
	}

	padZSize := []int{fft.dataSize[0], fft.dataSize[1], fft.fftSize[2] + 2}
	fft.padZ.Init(nComp, padZSize, DO_ALLOC)
}

func NewFFTPlan(nComp int, dataSize3D, fftSize3D []int) *FFTPlan{
	fft:=new(FFTPlan)
	fft.Init(nComp, dataSize3D, fftSize3D)
	return fft
}

func(fft*FFTPlan)Free(){
	fft.nComp=0
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.fftSize[i] = 0
	}
	(&(fft.padZ)).Free()

	// TODO destroy
}

func (fft *FFTPlan) Exec(in, out *Array) {
	fmt.Println("in:", in.LocalCopy().Array)
	CopyPadZ(&(fft.padZ), in)
	fmt.Println("padZ:", fft.padZ.LocalCopy().Array)

}
