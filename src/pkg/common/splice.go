//  This file is part of MuMax, a high-performance micromagnetic simulator.
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
	stream cuda.Stream // General-purpose stream for use with this slice (to avoid creating/destroying many streams)
}


// Allocates and initiates a new slice. See slice.Init().
func NewSlice(deviceId int, length int) *slice {
	s := new(slice)
	s.Init(deviceId, length)
	return s
}


// Initiates the slice to refer to an array of "length" float32s on GPU number "deviceId".
func (s *slice) Init(deviceId int, length int) {
	Assert(deviceId >= 0 && deviceId < cuda.GetDeviceCount())

	// Switch device context if necessary
	AssureDevice(deviceId)

	s.deviceId = deviceId
	s.stream = cuda.StreamCreate()
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
	(&(s.stream)).Destroy()
	s.deviceId = -1 // make sure it doesn't get used anymore
}




// A Splice represents distributed GPU memory in a transparent way.
type Splice struct {
	slice []slice
	length int
}


// See Splice.Init()
func NewSplice(length int) Splice {
	var s Splice
	s.Init(length)
	return s
}


// Initiates the Splice to represent "length" float32s,
// automatically distributed over all available GPUs.
func (s *Splice) Init(length int) {
	devices := getDevices()
	N := len(devices)
	Assert(length%N == 0)
	s.slice = make([]slice, N)
	for i := range devices {
		s.slice[i].Init(devices[i], length/N)
	}
	s.length = length
}


func (s *Splice) Len() int{
	return s.length
}


// Frees the underlying storage
func (s *Splice) Free() {
	for _, slice := range s.slice {
		slice.Free()
	}
}


// s = h
func (s *Splice) CopyFromHost(h []float32){
	Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i:= range s.slice{
		length := s.slice[i].array.Len()
		cuda.CopyFloat32ArrayToDevice(s.slice[i].array, h[start:start+length])
		start+=length
	}
}

// h = s
func (s *Splice) CopyToHost(h []float32){
	Assert(len(h) == s.Len()) // in principle redundant
	start := 0
	for i:= range s.slice{
		length := s.slice[i].array.Len()
		cuda.CopyDeviceToFloat32Array(h[start:start+length], s.slice[i].array)
		start+=length
	}
}

// d = s
//func (s *Splice) CopyToDevice(d Splice){
//	Assert(d.Len() == s.Len()) // in principle redundant
//	start := 0
//	for i:= range s.slice{
//		length := s.slice[i].array.Len()
//		cuda.CopyDeviceToDevice(d.slice[i].array, s.slice[i].array)
//		start+=length
//	}
//}


// Copy: s = d.
// The overall operation is synchronous but the underlying 
// copies on the separate devices overlap, effectively boosting
// the bandwidth by N for N devices.
func (s *Splice) CopyFromDevice(d Splice){
	Assert(d.Len() == s.Len()) // in principle redundant
	start := 0
	// Overlapping copies run concurrently on the individual devices
	for i:= range s.slice{
		length := s.slice[i].array.Len()
		cuda.CopyDeviceToDeviceAsync(s.slice[i].array, d.slice[i].array, s.slice[i].stream)
		start+=length
	}
	// Synchronize with all copies
	for _, sl := range s.slice{
		sl.stream.Synchronize()
	}
}


