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
	dataSize [3]int   // Size of the (non-zero) input data block
	kernSize [3]int   // Kernel size >= dataSize
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

var kernCompString map[int]string = map[int]string{XX: "XX", YY: "YY", ZZ: "ZZ", YZ: "YZ", XZ: "XZ", XY: "XY"}

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
		conv.kernSize[i] = kernSize[i]
	}

	Debug("ConvPlan.init", "dataSize:", dataSize, "kernSize:", kernSize)

	// init fftKern
	for i, k := range kernel {
		if k != nil {
			Debug("ConvPlan.init", "alloc:", kernCompString[i])
			conv.fftKern[i].Init(1, kernSize, DO_ALLOC)
		}
	}
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
