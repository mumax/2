//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files implements distributed memory over multiple GPUs
// When working with multiple GPUs, there is no notion of "current" device,
// hence these functions are allowed to change CUDAs current device (as returned by cuda.GetDevice())
// Author: Arne Vansteenkiste

package common

import (
	"cuda"
)


// Slices are the building blocks of Splices.
// A Slice resides on a single GPU. Multiple
// slices are combined into a splice.
type slice struct {
	array    *cuda.Float32Array // Access to the array on the GPU.
	deviceId int                // Identifies which GPU the array resides on.
}


// Allocates and initiates a new slice. See slice.Init().
func NewSlice(deviceId int, length int64) *slice {
	s := new(slice)
	s.Init(deviceId, length)
	return s
}


// Initiates the slice to refer to an array of "length" float32s on GPU number "deviceId".
func (s *slice) Init(deviceId int, length int64) {
	Assert(deviceId < cuda.GetDeviceCount())

	// Switch device context if necessary
	AssureDevice(deviceId)

	s.deviceId = deviceId
	s.array = cuda.NewFloat32Array(length)

}

func AssureDevice(deviceId int) (prevDevice int) {
	prevDevice = cuda.GetDevice()
	if prevDevice != deviceId {
		cuda.SetDevice(deviceId)
	}
	return
}

func (s *slice) Free() {
	// Switch device context if necessary
	AssureDevice(s.deviceId)
	s.array.Free()
}


// A Splice represents distributed GPU memory in a transparent way.
type Splice struct {
	slice []slice
}


// See Splice.Init()
func NewSplice(length int64) Splice {
	var s Splice
	s.Init(length)
	return s
}


// Initiates the Splice to represent "length" float32s,
// automatically distributed over all available GPUs.
func (s *Splice) Init(length int64) {
	devices := getDevices()
	N := int64(len(devices))
	Assert(length%N == 0)
	s.slice = make([]slice, N)
	for i := range devices {
		s.slice[i].Init(devices[i], length/N)
	}
}


func (s *Splice) Free() {
	for _, slice := range s.slice {
		slice.Free()
	}
}


type VSplice struct {
	Comp []Splice
	List Splice
}
