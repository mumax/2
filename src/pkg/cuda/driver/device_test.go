// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"testing"
	"fmt"
)

func TestDevice(t *testing.T) {
	fmt.Println("DeviceGetCount:", DeviceGetCount())
	for i := 0; i < DeviceGetCount(); i++ {
		fmt.Println("DeviceGet", i)
		dev := DeviceGet(i)
		major, minor := dev.ComputeCapability()
		fmt.Println("Name: ", dev.GetName())
		fmt.Println("ComputeCapability: ", major, minor)
		fmt.Println("TotalMem: ", dev.TotalMem())

		fmt.Println("ATTRIBUTE_MAX_THREADS_PER_BLOCK           :", dev.GetAttribute(ATTRIBUTE_MAX_THREADS_PER_BLOCK))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_X                 :", dev.GetAttribute(ATTRIBUTE_MAX_BLOCK_DIM_X))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_Y                 :", dev.GetAttribute(ATTRIBUTE_MAX_BLOCK_DIM_Y))
		fmt.Println("ATTRIBUTE_MAX_BLOCK_DIM_Z                 :", dev.GetAttribute(ATTRIBUTE_MAX_BLOCK_DIM_Z))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_X                  :", dev.GetAttribute(ATTRIBUTE_MAX_GRID_DIM_X))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_Y                  :", dev.GetAttribute(ATTRIBUTE_MAX_GRID_DIM_Y))
		fmt.Println("ATTRIBUTE_MAX_GRID_DIM_Z                  :", dev.GetAttribute(ATTRIBUTE_MAX_GRID_DIM_Z))
		fmt.Println("ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK     :", dev.GetAttribute(ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
		fmt.Println("ATTRIBUTE_TOTAL_CONSTANT_MEMORY           :", dev.GetAttribute(ATTRIBUTE_TOTAL_CONSTANT_MEMORY))
		fmt.Println("ATTRIBUTE_WARP_SIZE                       :", dev.GetAttribute(ATTRIBUTE_WARP_SIZE))
		fmt.Println("ATTRIBUTE_MAX_PITCH                       :", dev.GetAttribute(ATTRIBUTE_MAX_PITCH))
		fmt.Println("ATTRIBUTE_MAX_REGISTERS_PER_BLOCK         :", dev.GetAttribute(ATTRIBUTE_MAX_REGISTERS_PER_BLOCK))
		fmt.Println("ATTRIBUTE_CLOCK_RATE                      :", dev.GetAttribute(ATTRIBUTE_CLOCK_RATE))
		fmt.Println("ATTRIBUTE_TEXTURE_ALIGNMENT               :", dev.GetAttribute(ATTRIBUTE_TEXTURE_ALIGNMENT))
		fmt.Println("ATTRIBUTE_MULTIPROCESSOR_COUNT            :", dev.GetAttribute(ATTRIBUTE_MULTIPROCESSOR_COUNT))
		fmt.Println("ATTRIBUTE_KERNEL_EXEC_TIMEOUT             :", dev.GetAttribute(ATTRIBUTE_KERNEL_EXEC_TIMEOUT))
		fmt.Println("ATTRIBUTE_INTEGRATED                      :", dev.GetAttribute(ATTRIBUTE_INTEGRATED))
		fmt.Println("ATTRIBUTE_CAN_MAP_HOST_MEMORY             :", dev.GetAttribute(ATTRIBUTE_CAN_MAP_HOST_MEMORY))
		fmt.Println("ATTRIBUTE_COMPUTE_MODE                    :", dev.GetAttribute(ATTRIBUTE_COMPUTE_MODE))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH         :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH         :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT        :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH         :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT        :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH         :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS))
		fmt.Println("ATTRIBUTE_SURFACE_ALIGNMENT               :", dev.GetAttribute(ATTRIBUTE_SURFACE_ALIGNMENT))
		fmt.Println("ATTRIBUTE_CONCURRENT_KERNELS              :", dev.GetAttribute(ATTRIBUTE_CONCURRENT_KERNELS))
		fmt.Println("ATTRIBUTE_ECC_ENABLED                     :", dev.GetAttribute(ATTRIBUTE_ECC_ENABLED))
		fmt.Println("ATTRIBUTE_PCI_BUS_ID                      :", dev.GetAttribute(ATTRIBUTE_PCI_BUS_ID))
		fmt.Println("ATTRIBUTE_PCI_DEVICE_ID                   :", dev.GetAttribute(ATTRIBUTE_PCI_DEVICE_ID))
		fmt.Println("ATTRIBUTE_TCC_DRIVER                      :", dev.GetAttribute(ATTRIBUTE_TCC_DRIVER))
		fmt.Println("ATTRIBUTE_MEMORY_CLOCK_RATE               :", dev.GetAttribute(ATTRIBUTE_MEMORY_CLOCK_RATE))
		fmt.Println("ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH         :", dev.GetAttribute(ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH))
		fmt.Println("ATTRIBUTE_L2_CACHE_SIZE                   :", dev.GetAttribute(ATTRIBUTE_L2_CACHE_SIZE))
		fmt.Println("ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR  :", dev.GetAttribute(ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR))
		fmt.Println("ATTRIBUTE_ASYNC_ENGINE_COUNT              :", dev.GetAttribute(ATTRIBUTE_ASYNC_ENGINE_COUNT))
		fmt.Println("ATTRIBUTE_UNIFIED_ADDRESSING              :", dev.GetAttribute(ATTRIBUTE_UNIFIED_ADDRESSING))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH :", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH))
		fmt.Println("ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:", dev.GetAttribute(ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS))

		fmt.Printf("Properties:%#v\n", dev.GetProperties())
	}
}
