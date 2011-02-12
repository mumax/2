//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files implements distributed vector splices over multiple GPUs.
// A vector splice provides the ability to take a "component", which is 
// then a (scalar) splice.
// It is guaranteed that the component splices are again nicely distributed
// over the GPUs, which would not be case if one would allocate, e.g., a 3*N splice
// to represent N 3-vectors.
//
// Author: Arne Vansteenkiste

package common

import (
)


type VSplice struct {
	Comp []Splice // List of components, e.g. vector or tensor components
	List Splice   // All elements as a single, contiguous list.
}


func (v *VSplice) Init(components, length int) {
	v.List.Init(components * length)

	//devices := getDevices()
	//compSliceLen := distribute(length, devices) // length of slices in one component

	v.Comp = make([]Splice, components)
	for i,c := range v.Comp{
		c.length = length
		start := i * length
		stop := (i+1) * length
		c.slice = make([]slice, components)
		for j := range c.slice{
			cs := &(c.slice[j])
			cs.array.InitSlice(&(v.List.slice[j].array),start, stop)
		}
	}
}


func NewVSplice(components, length int) *VSplice{
	v := new(VSplice)
	v.Init(components, length)
	return v
}


func (v *VSplice) Free(){
	v.List.Free()
	v.Comp = nil
}
