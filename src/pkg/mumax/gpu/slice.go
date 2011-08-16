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
	//. "mumax/common"
	cu "cuda/driver"
	//cuda "cuda/runtime"
)


// Slices are the building blocks of Splices.
// A Slice resides on a single GPU. Multiple
// slices are combined into a splice.
type slice struct {
	array cu.DevicePtr // Access to the array on the GPU.
	//length int          // Number of floats
	//devId  int       // index of CUDA context of this slice's allocation
	stream cu.Stream // General-purpose stream for use with this slice (to avoid creating/destroying many streams)
}


// Allocates and initiates a new slice. See slice.Init().
//func newSlice(deviceId int, length int) *slice {
//	s := new(slice)
//	s.init(deviceId, length)
//	return s
//}


// Initiates the slice to refer to an array of "length" float32s on GPU number "deviceId".
//func (s *slice) init(devices_i int, slicelen int) {
//	//Assert(deviceId >= 0 && deviceId < cu.DeviceGetCount())
//
//	// Switch device context if necessary
//	assureContextId(devices_i)
//	s.devId = devices_i
//	s.array = cu.MemAlloc(SIZEOF_FLOAT * int64(slicelen))
//	s.stream = cu.StreamCreate()
//	s.length = slicelen
//}


// Takes a sub-slice.
//func (b *slice) initSlice(a *slice, start, stop int) {
//	if b.array != cu.DevicePtr(uintptr(0)) {
//		panic("cuda slice already initialized")
//	}
//	assureContextId(a.devId)
//	b.array = cu.DevicePtr(offset(uintptr(a.array), start*SIZEOF_FLOAT))
//	b.length = stop - start
//	b.devId = a.devId
//	b.stream = cu.StreamCreate()
//}

// Pointer arithmetic.
func offset(ptr uintptr, bytes int) uintptr {
	return ptr + uintptr(bytes)
}


//func (s *slice) free() {
//	assureContextId(s.devId) // necessary in a multi-GPU context
//	s.array.Free()
//	s.stream.Destroy()
//	s.devId = -1 // invalid id to make sure it's not used
//}

func sliceFree(devId int, array cu.DevicePtr, stream cu.Stream) {
	assureContextId(devId) // necessary in a multi-GPU context
	array.Free()
	stream.Destroy()
}


//func (s *slice) Len() int {
//	return s.length
//}
