//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implents 3-dimensional arrays of N-vectors on the GPU
// Author: Arne Vansteenkiste

import ()


// A MuMax Array represents a 3-dimensional array of N-vectors.
type Array struct {
	splice vSplice // Underlying multi-GPU storage
	_size  [4]int  // INTERNAL {components, size0, size1, size2}
	size4D []int   // {components, size0, size1, size2}
	size3D []int   // {size0, size1, size2}
	length int
}


func (t *Array) Init(components int, size3D []int) {
	Assert(len(size3D) == 3)
	t.length = Prod(size3D)
	t.splice.Init(components, t.length)
	t._size[0] = components
	for i := range size3D {
		t._size[i+1] = size3D[i]
	}
	t.size4D = t._size[:]
	t.size3D = t._size[1:]
}


func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D)
	return t
}


func (t *Array) Free() {
	t.splice.Free()
	for i := range t._size {
		t._size[i] = 0
	}
	t.length = 0
}

func (a *Array) IsNil() bool {
	return a.splice.IsNil()
}

func (dst *Array) CopyFromDevice(src *Array) {
	// test for equal size
	for i, d := range dst._size {
		if d != src._size[i] {
			panic(MSG_ARRAY_SIZE_MISMATCH)
		}
	}
	(&(dst.splice)).CopyFromDevice(&(src.splice))
}


const MSG_ARRAY_SIZE_MISMATCH = "array size mismatch"
