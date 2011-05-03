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

package common

import (
	cu "cuda/driver"
	//"fmt"
)


// Slices are the building blocks of Splices.
// A Slice resides on a single GPU. Multiple
// slices are combined into a splice.
type slice struct {
	array  cu.DevicePtr // Access to the array on the GPU.
	length int          // Number of floats
	devId  int          // index of CUDA context of this slice's allocation
	stream cu.Stream    // General-purpose stream for use with this slice (to avoid creating/destroying many streams)
}


// Allocates and initiates a new slice. See slice.Init().
func newSlice(deviceId int, length int) *slice {
	s := new(slice)
	s.init(deviceId, length)
	return s
}


// Initiates the slice to refer to an array of "length" float32s on GPU number "deviceId".
func (s *slice) init(deviceId int, length int) {
	Assert(deviceId >= 0 && deviceId < cu.DeviceGetCount())

	// Switch device context if necessary
	assureContext(getDeviceContext(deviceId))

	s.devId = deviceId
	s.stream = cu.StreamCreate()
	s.array = cu.MemAlloc(SIZEOF_FLOAT * int64(length))
	s.length = length
}


// Takes a sub-slice.
func (b *slice) initSlice(a *slice, start, stop int) {
	if b.array != cu.DevicePtr(uintptr(0)) {
		panic("cuda slice already initialized")
	}
	assureContextId(a.devId)
	b.array = cu.DevicePtr(offset(uintptr(a.array), start*SIZEOF_FLOAT))
	b.length = stop - start
	b.devId = a.devId
	b.stream = cu.StreamCreate()
}

// Pointer arithmetic.
func offset(ptr uintptr, bytes int) uintptr {
	return ptr + uintptr(bytes)
}


func (s *slice) free() {
	//assureContext(s.ctx) // seems not necessary.
	s.array.Free()
	s.stream.Destroy()
	s.devId = -1 // invalid id to make sure it's not used
}


func (s *slice) Len() int {
	return s.length
}


// A Splice represents distributed GPU memory in a transparent way.
type splice struct {
	slice  []slice // Arrays on different GPUs, each holding a part of the data
	length int     // Total number of float32s in the splice
}

//func (s *splice) String() string {
//	str := "splice{" +
//		"len=" + fmt.Sprint(s.length)
//	for i := range s.slice {
//		str += " " + s.slice[i].String()
//	}
//	str += "}"
//	return str
//}

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
	s.length = length
}


// Distributes elements over the available GPUs.
// length: number of elements to distribute.
// slicelen[i]: number of elements for device i.
// TODO(a) Slicer
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
	return s.length
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
