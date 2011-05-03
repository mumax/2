//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implents 3-dimensional arrays of N-vectors on the host.
// Author: Arne Vansteenkiste

import (
)


// A MuMax Array represents a 3-dimensional array of N-vectors.
type HostArray struct {
	list []float32 // Underlying storage
	array [][][][]float32
	_size  [4]int  // INTERNAL {components, size0, size1, size2}
	size4D []int   // {components, size0, size1, size2}
	size3D []int   // {size0, size1, size2}
}


func (t *HostArray) Init(components int, size3D []int) {
	Assert(len(size3D) == 3)

	t._size[0] = components
	for i := range size3D {
		t._size[i+1] = size3D[i]
	}
	t.size4D = t._size[:]
	t.size3D = t._size[1:]
	t.list, t.array = Array4D(t._size[0], t._size[1], t._size[2], t._size[3])
}


func NewHostArray(components int, size3D []int) *HostArray {
	t := new(HostArray)
	t.Init(components, size3D)
	return t
}


