//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package host

// This file implements 3-dimensional arrays of N-vectors on the host.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"sync"
)


// A MuMax Array represents a 3-dimensional array of N-vectors.
type Array struct {
	List         []float32       // Underlying contiguous storage
	Array        [][][][]float32 // Array in the usual way
	Comp         [][]float32     // Components as contiguous lists
	Size         [4]int          // INTERNAL {components, size0, size1, size2}
	Size4D       []int           // {components, size0, size1, size2}
	Size3D       []int           // {size0, size1, size2}
	sync.RWMutex                 // mutex for safe concurrent access to this array
}


// Initializes a pre-allocated Array struct
func (t *Array) Init(components int, size3D []int) {
	Assert(len(size3D) == 3)
	t.List, t.Array = Array4D(components, size3D[0], size3D[1], size3D[2])
	t.Comp = Slice2D(t.List, []int{components, Prod(size3D)})
	t.Size[0] = components
	t.Size[1] = size3D[0]
	t.Size[2] = size3D[1]
	t.Size[3] = size3D[2]
	t.Size4D = t.Size[:]
	t.Size3D = t.Size[1:]
}


// Allocates an returns a new Array
func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D)
	return t
}

func (a *Array) Rank() int {
	return len(a.Size)
}

func (a *Array) Len() int {
	return a.Size[0] * a.Size[1] * a.Size[2] * a.Size[3]
}
