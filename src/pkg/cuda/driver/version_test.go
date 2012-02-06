// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"testing"
	"fmt"
)

func TestVersion(t *testing.T) {
	fmt.Println("CUDA driver version: ", GetVersion())
	//	for i := 0; i < DeviceGetCount(); i++ {
	//		dev := DeviceGet(i)
	//		major, minor := dev.ComputeCapability()
	//		fmt.Println("CUDA device", i, "compute capability: ", major, minor)
	//		fmt.Println("CUDA device", i, "total memory: ", dev.TotalMem())
	//	}
}
