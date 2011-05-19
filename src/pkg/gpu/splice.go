//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files implements distributed memory over multiple GPUs
// When working with multiple GPUs, there is no notion of "current" device,
// hence these functions are allowed to change CUDA's current device (as returned by cuda.GetDevice())
// Author: Arne Vansteenkiste

package gpu

import (
	. "mumax/common"
	cu "cuda/driver"
)


// A Splice represents distributed GPU memory in a transparent way.
type splice struct {
	slice  []slice // Arrays on different GPUs, each holding a part of the data
	//length int     // Total number of float32s in the splice
}


// See Splice.Init()
func newSplice(length int) splice {
	var s splice
	s.init(length)
	return s
}


// Initiates the splice to represent "length" float32s,
// automatically distributed over all available GPUs.
func (s *splice) init(length int) {
	devices := getDevices()
	s.slice = make([]slice, len(devices))
	slicelen := distribute(length, devices)
	for i := range devices {
		s.slice[i].init(devices[i], slicelen[i])
	}
	//s.length = length
}


// Distributes elements over the available GPUs.
// length: number of elements to distribute.
// slicelen[i]: number of elements for device i.
func distribute(length int, devices []int) (slicelen []int) {
	N := len(devices)
	slicelen = make([]int, N)

	// For now: equal slicing
	Assert(length%N == 0)
	for i := range slicelen {
		slicelen[i] = length / N
	}
	return
}


// Total number of float32 elements.
func (s *splice) Len() int {
	l := 0
	for i := range s.slice{
		l += s.slice[i].length
	}
	return l
}


// Frees the underlying storage
func (s *splice) Free() {
	for i := range s.slice {
		(&(s.slice[i])).free()
	}
}


func (s *splice) IsNil() bool {
	if s.slice == nil {
		return true
	}
	return s.slice[0].array == cu.DevicePtr(0)
}

// s = h.
// TODO(a) Could be overlapping
func (s *splice) CopyFromHost(h []float32) {
	Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i := range s.slice {
		length := s.slice[i].length
		cu.MemcpyHtoD(cu.DevicePtr(s.slice[i].array), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
		start += length
	}
}

// h = s.
// TODO(a) Could be overlapping
func (s *splice) CopyToHost(h []float32) {
	Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i := range s.slice {
		length := s.slice[i].length
		cu.MemcpyDtoH(cu.HostPtr(&h[start]), cu.DevicePtr(s.slice[i].array), SIZEOF_FLOAT*int64(length))
		start += length
	}
}

// Copy: s = d.
// The overall operation is synchronous but the underlying 
// copies on the separate devices overlap, effectively boosting
// the bandwidth by N for N devices.
func (s *splice) CopyFromDevice(d splice) {
	Assert(d.Len() == s.Len()) // in principle redundant
	start := 0
	// copies run concurrently on the individual devices
	for i := range s.slice {
		length := s.slice[i].length // in principle redundant
		Assert(length == d.slice[i].length)
		cu.MemcpyDtoDAsync(cu.DevicePtr(s.slice[i].array), cu.DevicePtr(d.slice[i].array), SIZEOF_FLOAT*int64(length), s.slice[i].stream)
		start += length
	}
	// Synchronize with all copies
	for i := range s.slice {
		s.slice[i].stream.Synchronize()
	}
}
