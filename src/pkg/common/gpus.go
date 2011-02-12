//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file implements GPU selection for multi-device operation.
// Author: Arne Vansteenkiste

package common

import (
	"fmt"
	"cuda"
)


// INTERNAL: List of GPU ids to use for multi-GPU operation.
var _useDevice []int = nil


// Returns the list of usable devices. 
func getDevices() []int {
	if _useDevice == nil {
		panic(Bug(MSG_DEVICEUNINITIATED))
	}
	return _useDevice
}


// Sets a list of devices to use.
func UseDevice(devices []int) {
	Assert(_useDevice == nil)
	N := cuda.GetDeviceCount()
	for _, n := range devices {
		if n >= N {
			panic(InputErr(MSG_BADDEVICEID + fmt.Sprint(n)))
		}
	}
	copy(_useDevice, devices)
}


// Uses all available GPUs
func UseAllDevices() {
	Assert(_useDevice == nil)
	N := cuda.GetDeviceCount()
	for i := 0; i < N; i++ {
		_useDevice = append(_useDevice, i)
	}
}


// Use GPU list suitable for debugging:
// if only 1 is present: use it twice to
// test "multi"-GPU code. The distribution over
// two separate arrays on the same device is a bit
// silly, but good for debugging.
func UseDebugDevices() {
	Assert(_useDevice == nil)
	N := cuda.GetDeviceCount()
	for i := 0; i < N; i++ {
		_useDevice = append(_useDevice, i)
	}
	if N == 1 {
		_useDevice = append(_useDevice, 0) // Use the same device twice.
	}
}


// Error message
const (
	MSG_BADDEVICEID       = "Invalid device ID: "
	MSG_DEVICEUNINITIATED = "Device list not initiated"
)
