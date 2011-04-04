//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import ()


type DevTensor struct {
	splice VSplice // Underlying multi-GPU storage
	_size  [4]int  // INTERNAL {components, size0, size1, size2}
	size4D []int   // {components, size0, size1, size2}
	size3D []int   // {size0, size1, size2}
	length int
}


func (t *DevTensor) Init(components int, size3D []int) {
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


func NewDevTensor(components int, size3D []int) *DevTensor {
	t := new(DevTensor)
	t.Init(components, size3D)
	return t
}


func (t *DevTensor) Free() {
	t.splice.Free()
	for i := range t._size {
		t._size[i] = 0
	}
	t.length = 0
}


func (dst *DevTensor) CopyFromDevice(src *DevTensor){
	// test for equal size
	for i,d := range dst._size{
		if d != src._size[i]{
			panic(MSG_TENSOR_SIZE_MISMATCH)
		}
	}
	(&(dst.splice)).CopyFromDevice(&(src.splice))
}




const MSG_TENSOR_SIZE_MISMATCH = "tensor size mismatch"

func Prod(a []int) int {
	p := 1
	for _, x := range a {
		p *= x
	}
	return p
}
