//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import ()

// The default FFT constructor.
// The function pointer may be changed 
// to use a different FFT implementation globally.
var NewDefaultFFT func(dataSize, logicSize []int) FFTInterface = NewFFTPlan1

// Global map with all registered FFT plans
var fftPlans map[string]func(dataSize, logicSize []int) FFTInterface

// Interface for any sparse FFT plan.
type FFTInterface interface {
	Forward(in, out *Array)
	Inverse(in, out *Array)
	Free()
}

// Returns the normalization factor of an FFT with this logic size.
// (just the product of the sizes)
func FFTNormLogic(logicSize []int) int {
	return (logicSize[0] * logicSize[1] * logicSize[2])
}

// Returns the (NDevice-dependent) output size of an FFT with given logic size.
func FFTOutputSize(logicSize []int) []int {
  
  outputSize := make([]int, 3)
  outputSize[0] = logicSize[0]
  if (NDevice()==1){
    outputSize[1] = logicSize[1]
    outputSize[2] = logicSize[2] + 2 // One extra row of complex numbers
  } else{ //multi-gpu: YZ-transposed output!!
    outputSize[1] = logicSize[2] + 2*NDevice() // One extra row of complex numbers PER GPU
    outputSize[2] = logicSize[1]
  }
  
  return outputSize
}
