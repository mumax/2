//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements 3-dimensional arrays of N-vectors on the host.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
)


// A MuMax Array represents a 3-dimensional array of N-vectors.
type HostArray struct {
	List  []float32 // Underlying storage
	Array [][][][]float32
	Comp  [][]float32
}


func (t *HostArray) Init(components int, size3D []int) {
	Assert(len(size3D) == 3)
	t.List, t.Array = Array4D(components, size3D[0], size3D[1], size3D[2])
	t.Comp = Slice2D(t.List, []int{components, Prod(size3D)})
}


func NewHostArray(components int, size3D []int) *HostArray {
	t := new(HostArray)
	t.Init(components, size3D)
	return t
}
