//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements GPU selection for multi-device operation.
// Author: Arne Vansteenkiste

//#include "libmumax2.h"
import "C"

import (
	. "mumax/common"
	cu "cuda/driver"
	cuda "cuda/runtime"
	"unsafe"
	"fmt"
)

// INTERNAL: List of GPU ids to use for multi-GPU operation. E.g.: {0,1,2,3}
var _useDevice []int = nil

// INTERNAL: Device properties
var (
	maxThreadsPerBlock int
	maxBlockDim        [3]int
	maxGridDim         [3]int
)

// Sets a list of devices to use.
func InitMultiGPU(devices []int, flags uint) {
	Debug("InitMultiGPU ", devices, flags)

	initMultiGPUList(devices)
	initMultiGPUCgo()
	printMultiGPUInfo()
	initMultiGPUProperties()
	initMultiGPUPeerAccess()

	stream0 := make([]cu.Stream, NDevice())
	STREAM0 = Stream(stream0)
}

// init global _useDevice
func initMultiGPUList(devices []int) {
	Assert(len(devices) > 0)
	Assert(_useDevice == nil) // should not yet be initialized

	// check if device ID's are valid GPU numbers
	N := cu.DeviceGetCount()
	for _, n := range devices {
		if n >= N || n < 0 {
			panic(InputErr(MSG_BADDEVICEID + fmt.Sprint(n)))
		}
	}

	// set up list with used device IDs
	_useDevice = make([]int, len(devices))
	copy(_useDevice, devices)
}

// set C global variables. refer libmumax2,global.h
func initMultiGPUCgo() {
	Debug("setUsedDevices", _useDevice, len(_useDevice))
	C.setUsedGPUs((*C.int)(unsafe.Pointer(&_useDevice[0])), C.int(len(_useDevice)))
}

// output device info
func printMultiGPUInfo() {
	for i := range _useDevice {
		dev := cu.DeviceGet(_useDevice[i])
		Log("device", i, "( PCI", dev.GetAttribute(cu.A_PCI_DEVICE_ID), ")", dev.GetName(), ",", dev.TotalMem()/(1024*1024), "MiB")
	}

}

// set up device properties
func initMultiGPUProperties() {
	dev := cu.DeviceGet(_useDevice[0])
	maxThreadsPerBlock = dev.GetAttribute(cu.A_MAX_THREADS_PER_BLOCK)
	maxBlockDim[0] = dev.GetAttribute(cu.A_MAX_BLOCK_DIM_X)
	maxBlockDim[1] = dev.GetAttribute(cu.A_MAX_BLOCK_DIM_Y)
	maxBlockDim[2] = dev.GetAttribute(cu.A_MAX_BLOCK_DIM_Z)
	maxGridDim[0] = dev.GetAttribute(cu.A_MAX_GRID_DIM_X)
	maxGridDim[1] = dev.GetAttribute(cu.A_MAX_GRID_DIM_Y)
	maxGridDim[2] = dev.GetAttribute(cu.A_MAX_GRID_DIM_Z)
	Debug("Max", maxThreadsPerBlock, "threads per block, max", maxGridDim, "x", maxBlockDim, "threads per GPU")
}

// init inter-device access
func initMultiGPUPeerAccess() {
	// first init contexts
	for i := range _useDevice {
		setDevice(_useDevice[i])
		dummy := cuda.Malloc(1) // initializes a cuda context for the device
		cuda.Free(dummy)
	}

	// enable peer access if more than 1 GPU is specified
	// do not try to enable for one GPU so that device with CC < 2.0
	// can still be used in a single-GPU setup
	// also do not enable if GPU 0 is used twice for debug purposes
	if len(_useDevice) > 1 && !allZero(_useDevice) {
		Debug("Enabling device peer-to-peer access")
		for i := range _useDevice { //_deviceCtxs {
			//dev := cu.DeviceGet(_useDevice[i])
			//Debug("Device ", i, "UNIFIED_ADDRESSING:", dev.GetAttribute(cu.A_UNIFIED_ADDRESSING))
			//if dev.GetAttribute(cu.A_UNIFIED_ADDRESSING) != 1 {
			//	panic(ERR_UNIFIED_ADDR)
			//}
			for j := range _useDevice {
				//Debug("CanAccessPeer", i, j, ":", cu.DeviceCanAccessPeer(cu.DeviceGet(_useDevice[i]), cu.DeviceGet(_useDevice[j])))
				if i != j {
					//if !cu.DeviceCanAccessPeer(cu.DeviceGet(_useDevice[i]), cu.DeviceGet(_useDevice[j])) {
					//	panic(ERR_UNIFIED_ADDR)
					//}
					// enable access between context i and j
					//_deviceCtxs[i].SetCurrent()
					//_deviceCtxs[j].EnablePeerAccess()
					cuda.SetDevice(_useDevice[i])
					cuda.DeviceEnablePeerAccess(_useDevice[j])
				}
			}
		}
	}

	// set the current context
	cuda.SetDevice(_useDevice[0])
}

// Error message
const ERR_UNIFIED_ADDR = "A GPU does not support unified addressing and can not be used in a multi-GPU setup."

// INTERNAL: true if all elements are 0.
func allZero(a []int) bool {
	for _, n := range a {
		if n != 0 {
			return false
		}
	}
	return true
}

// Like InitMultiGPU(), but uses all available GPUs.
func InitAllGPUs(flags uint) {
	var use []int
	N := cu.DeviceGetCount()
	for i := 0; i < N; i++ {
		use = append(use, i)
	}
	InitMultiGPU(use, flags)
}

// Use GPU list suitable for debugging:
// if only 1 is present: use it twice to
// test "multi"-GPU code. The distribution over
// two separate arrays on the same device is a bit
// silly, but good for debugging.
func InitDebugGPUs() {
	var use []int
	N := cu.DeviceGetCount()
//   N := 1 
	for i := 0; i < N; i++ {
		use = append(use, i)
	}
	if N == 1 {
		println("WARNING: using only one GPU")
		// 		use = append(use, 0) // Use the same device twice.
	}
	InitMultiGPU(use, 0)
}

// Assures Context ctx[id] is currently active. Switches contexts only when necessary.
func setDevice(deviceId int) {
	// debug: test if device is supposed to be used
	ok := false
	for _, d := range _useDevice {
		if deviceId == d {
			ok = true
			break
		}
	}
	if !ok {
		panic(Bug(fmt.Sprint("Invalid device Id", deviceId, "should be in", _useDevice)))
	}

	// actually set the device
	cuda.SetDevice(deviceId)
}

func SetDeviceForIndex(index int) {
	setDevice(_useDevice[index])
}

// Returns the list of usable devices. 
func getDevices() []int {
	if _useDevice == nil {
		panic(Bug(MSG_DEVICEUNINITIATED))
	}
	return _useDevice
}

// Returns the number of used GPUs.
// This may be less than the number of available GPUs.
// (or even more)
func NDevice() int {
	return len(_useDevice)
}

// Error message
const (
	MSG_BADDEVICEID       = "Invalid device ID: "
	MSG_DEVICEUNINITIATED = "Device list not initiated"
)

// Stream 0 on each GPU
var STREAM0 Stream
