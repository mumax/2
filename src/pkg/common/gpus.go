//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements GPU selection for multi-device operation.
// Author: Arne Vansteenkiste

import (
	"fmt"
	cu "cuda/driver"
)


// INTERNAL: List of GPU ids to use for multi-GPU operation.
var _useDevice []int = nil

// INTERNAL: List of contexts for each used device.
var _deviceCtxs []cu.Context

// INTERNAL: The current CUDA context
var _currentCtx cu.Context


// Sets a list of devices to use.
func InitMultiGPU(devices []int, flags uint) {
	Assert(_useDevice == nil)
	N := cu.DeviceGetCount()
	for _, n := range devices {
		if n >= N || n < 0 {
			panic(InputErr(MSG_BADDEVICEID + fmt.Sprint(n)))
		}
	}
	copy(_useDevice, devices)

	// setup contexts
	_deviceCtxs = make([]cu.Context, len(devices))
	for i := range _deviceCtxs {
		_deviceCtxs[i] = cu.CtxCreate(flags, cu.DeviceGet(_useDevice[i]))
	}
	_currentCtx = _deviceCtxs[0]
}


// Uses all available GPUs
func InitAllGPU(flags uint) {
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
func InitDebugGPU() {
	var use []int
	N := cu.DeviceGetCount()
	for i := 0; i < N; i++ {
		use = append(use, i)
	}
	if N == 1 {
		use = append(use, 0) // Use the same device twice.
	}
	InitMultiGPU(use, 0)
}

// Assures Context ctx is currently active. Switches contexts only when necessary.
func assureContext(ctx cu.Context) {
	if _currentCtx != ctx {
		ctx.SetCurrent()
		_currentCtx = ctx
	}
}

// Returns the current context
func getContext() cu.Context{
	return _currentCtx
}

// Returns the list of usable devices. 
func getDevices() []int {
	if _useDevice == nil {
		panic(Bug(MSG_DEVICEUNINITIATED))
	}
	return _useDevice
}


// Returns a context for the current device.
func getDeviceContext(deviceId int) cu.Context {
	return _deviceCtxs[deviceId]
}


// Error message
const (
	MSG_BADDEVICEID       = "Invalid device ID: "
	MSG_DEVICEUNINITIATED = "Device list not initiated"
)
