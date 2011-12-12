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
