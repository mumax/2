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
//	cu "cuda/driver"
//	"cuda/cufft"
//	"fmt"
)

type ConvPlan struct {
	dataSize [3]int         // Size of the (non-zero) input data block
	kernSize  [3]int        // Kernel size >= dataSize
	kernel [9]*Array	 //kernel components 
}

func (conv *ConvPlan) Init(dataSize, kernSize []int) {
	Assert(len(dataSize) == 3)
	Assert(len(kernSize) == 3)
	//NDev := NDevice()

	// init size
	for i := range conv.dataSize {
		conv.dataSize[i] = dataSize[i]
		conv.kernSize[i] = kernSize[i]
	}
}

func NewConvPlan(dataSize, kernSize []int) *ConvPlan {
	conv := new(ConvPlan)
	conv.Init(dataSize, kernSize)
	return conv
}

func (conv *ConvPlan) Free() {
	// TODO
}


func (conv *ConvPlan) Convolve(in, out *Array){

}



