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

// Layout example for a (3,4) vsplice on 2 GPUs:
// GPU0: X0 X1  Y0 Y1 Z0 Z1
// GPU1: X2 X3  Y2 Y3 Z2 Z3
//
type VSplice struct {
	Comp []Splice // List of components, e.g. vector or tensor components
	list Splice   // All elements as a single, contiguous list. The memory layout is not simple enough for a host array to be directly copied to it.
}


func (v *VSplice) Init(components, length int) {
	v.list.Init(components * length)

	devices := getDevices()
	Ndev := len(devices)
	compSliceLen := distribute(length, devices) 

	v.Comp = make([]Splice, components)
	c := v.Comp
	for i := range v.Comp{
		c[i].length = length
		c[i].slice = make([]slice, Ndev)
		for j := range c[i].slice{
			cs := &(c[i].slice[j])
			start := i * compSliceLen[j]
			stop := (i+1) * compSliceLen[j]
			cs.InitSlice(&(v.list.slice[j]),start, stop)
		}
	}
}


func NewVSplice(components, length int) *VSplice{
	v := new(VSplice)
	v.Init(components, length)
	return v
}


func (v *VSplice) Free(){
	v.list.Free()
	//TODO(a) Destroy streams.
	// nil pointers, zero lengths, just to be sture
	v.Comp = nil
}


func (dst *VSplice) CopyFromDevice(src *VSplice){
	dst.list.CopyFromDevice(src.list)
}

//func (src *VSplice) CopyToDevice(dst *VSplice){
//	src.list.CopyToDevice(dst.list)
//}
