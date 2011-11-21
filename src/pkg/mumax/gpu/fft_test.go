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
	"fmt"
)

func TestFFT(test *testing.T) {
	fmt.Println("FFT Test")
	nComp := 1
	N0, N1, N2 := 4, 8, 8
	dataSize := []int{N0, N1, N2}
	fftSize := []int{N0, N1, N2}
	fft := NewFFTPlan(dataSize, fftSize)
	defer fft.Free()

	in := NewArray(nComp, dataSize)
	defer in.Free()
	out := NewArray(nComp, []int{fftSize[0], fftSize[1], fftSize[2] + 2})
	inh := in.LocalCopy()

	//a := inh.Array[0]
	//n := 0
	//for i := 0; i < N0; i++ {
	//	n  = 0
	//	for j := 0; j < N1; j++ {
	//		for k := 0; k < N2; k++ {
	//			if i == 0 {
	//				a[i][j][k] = float32(n)
	//			}
	//			n++
	//		}
	//	}
	//}
	inh.List[0] = 1

	//inh.List[0] = 1
	in.CopyFromHost(inh)

	fft.Forward(in, out)

}

func BenchmarkFFT(b *testing.B) {
	b.StopTimer()

	nComp := 1
	N := 2048
	N0, N1, N2 := 1, N, N
	dataSize := []int{N0, N1, N2}
	fftSize := []int{N0, N1, N2}
	fft := NewFFTPlan(dataSize, fftSize)
	defer fft.Free()

	in := NewArray(nComp, dataSize)
	defer in.Free()

	// warmup
	fft.Forward(in, nil)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		fft.Forward(in, nil)
	}

}
