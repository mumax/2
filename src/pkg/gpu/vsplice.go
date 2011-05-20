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

package gpu

import (
	. "mumax/common"
	cu "cuda/driver"
)

// Layout example for a (3,4) vsplice on 2 GPUs:
// GPU0: X0 X1  Y0 Y1 Z0 Z1
// GPU1: X2 X3  Y2 Y3 Z2 Z3
//
type vSplice struct {
	Comp [][]slice // List of components, e.g. vector or tensor components
	list []slice   // All elements as a single, contiguous list. The memory layout is not simple enough for a host array to be directly copied to it.
}


// Initializes a Vector Splice to hold length * components float32s.
// E.g.: Init(3, 1000) gives an array of 1000 3-vectors
// E.g.: Init(1, 1000) gives an array of 1000 scalars
// E.g.: Init(6, 1000) gives an array of 1000 6-vectors or symmetric tensors
func (v *vSplice) InitVSplice(components, length int) {
	Assert(components > 0)

	devices := getDevices()
	v.list = splice(make([]slice, len(devices)))
	slicelen := distribute(components*length, devices)
	for i := range devices {
		v.list[i].init(devices[i], slicelen[i])
	}

	Ndev := len(devices)
	compSliceLen := distribute(length, devices)

	v.Comp = make([][]slice, components)
	//c := v.Comp
	for i := range v.Comp {
		//c[i].length = length
		v.Comp[i] = splice(make([]slice, Ndev))
		for j := range v.Comp[i] {
			cs := &(v.Comp[i][j])
			start := i * compSliceLen[j]
			stop := (i + 1) * compSliceLen[j]
			cs.initSlice(&(v.list[j]), start, stop)
		}
	}
}


// Allocates a new Vector Splice.
// See Init()
func newVSplice(components, length int) *vSplice {
	v := new(vSplice)
	v.InitVSplice(components, length)
	return v
}


// Frees the Vector Splice.
// This makes the Component Splices unusable.
func (v *vSplice) FreeVSplice() {
	//v.list.Free()
	for i := range v.list {
		(&(v.list[i])).free()
	}

	//TODO(a) Destroy streams.
	// nil pointers, zero lengths, just to be sture
	for i := range v.Comp {
		slice := v.Comp[i]
		for j := range slice {
			// The slice must not be freed because the underlying list has already been freed.
			slice[j].devId = -1 // invalid id 
			slice[j].stream.Destroy()
		}
	}
	v.Comp = nil
}


func (v *vSplice) IsNil() bool {
	return v.list == nil
}

// Total number of float32 elements.
//func (v *vSplice) Len() int {
//	return v.list.length
//}


// Number of components.
func (v *vSplice) NComp() int {
	return len(v.Comp)
}


// returns {NComp(), Len()/NComp()}
func (v *vSplice) Size() [2]int {
	return [2]int{len(v.Comp), len(v.Comp[0])}
}


func (dst *vSplice) VSpliceCopyFromDevice(src *vSplice) {
	//dst.list.CopyFromDevice(src.list)
	d := dst.list
	s := src.list
	Assert(len(d) == len(s)) // in principle redundant
	start := 0
	// copies run concurrently on the individual devices
	for i := range s {
		length := s[i].length // in principle redundant
		Assert(length == d[i].length)
		cu.MemcpyDtoDAsync(cu.DevicePtr(s[i].array), cu.DevicePtr(d[i].array), SIZEOF_FLOAT*int64(length), s[i].stream)
		start += length
	}
	// Synchronize with all copies
	for i := range s {
		s[i].stream.Synchronize()
	}

}

//func (src *VSplice) CopyToDevice(dst *VSplice){
//	src.list.CopyToDevice(dst.list)
//}


func (dst *vSplice) VSpliceCopyFromHost(src [][]float32) {
	Assert(dst.NComp() == len(src))
	// we have to work component-wise because of the data layout on the devices
	for i := range src {
		//Assert(len(dst.Comp[i]) == len(src[i])) // TODO(a): redundant
		//dst.Comp[i].CopyFromHost(src[i])

		h := src[i]
		s := dst.Comp[i]
		//Assert(len(h) == len(s)) // in principle redundant
		start := 0
		for i := range s {
			length := s[i].length
			cu.MemcpyHtoD(cu.DevicePtr(s[i].array), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
			start += length
		}

	}
}


func (src *vSplice) VSpliceCopyToHost(dst [][]float32) {
	Assert(src.NComp() == len(dst))
	for i := range dst {
		//Assert(len(src.Comp[i]) == len(dst[i])) // TODO(a): redundant
		//src.Comp[i].CopyToHost(dst[i])

	h := dst[i]
	s := src.Comp[i]
	//Assert(len(h) == len(s)) // in principle redundant
	start := 0
	for i := range s {
		length := s[i].length
		cu.MemcpyDtoH(cu.HostPtr(&h[start]), cu.DevicePtr(s[i].array), SIZEOF_FLOAT*int64(length))
		start += length
	}


	}
}
