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
type splice []slice // Arrays on different GPUs, each holding a part of the data






// Total number of float32 elements.
//func (s splice) Len() int {
//	l := 0
//	for i := range s {
//		l += s[i].length
//	}
//	return l
//}


// Frees the underlying storage
func (s splice) Free() {
	for i := range s {
		(&(s[i])).free()
	}
}


func (s splice) IsNil() bool {
	if s == nil {
		return true
	}
	return s[0].array == cu.DevicePtr(0)
}

// s = h.
// TODO(a) Could be overlapping
func (s splice) CopyFromHost(h []float32) {
	//Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i := range s {
		length := s[i].length
		cu.MemcpyHtoD(cu.DevicePtr(s[i].array), cu.HostPtr(&h[start]), SIZEOF_FLOAT*int64(length))
		start += length
	}
}

// h = s.
// TODO(a) Could be overlapping
func (s splice) CopyToHost(h []float32) {
	//Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i := range s {
		length := s[i].length
		cu.MemcpyDtoH(cu.HostPtr(&h[start]), cu.DevicePtr(s[i].array), SIZEOF_FLOAT*int64(length))
		start += length
	}
}

// Copy: s = d.
// The overall operation is synchronous but the underlying 
// copies on the separate devices overlap, effectively boosting
// the bandwidth by N for N devices.
func (s splice) CopyFromDevice(d splice) {
	//Assert(d.Len() == s.Len()) // in principle redundant
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
